import os
import torch
import bitsandbytes as bnb
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaRMSNorm
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import logging
import math
import types
import torch.nn.functional as F

# ----------------------
# Critical Configuration
# ----------------------
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"  # Enable verbose logging for torch.compile
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"  # Reduce compilation threads
os.environ["TORCH_COMPILE_DEBUG"] = "1"  # Enable debug mode for torch.compile
os.environ["TORCH_LOGS"] = "+dynamo,+aot,+inductor"
os.environ["TORCHDYNAMO_REPORT_GUARD_FAILURES"] = "1"
os.environ["TORCHDYNAMO_COMPILE_VERBOSE"] = "1"
os.environ["TORCH_DYNAMO_EXPORT_AOT_GRAPH"] = "1"  # Export AOT graphs for debugging

# ---------------------
# BnB 4-bit Patching
# ---------------------
def dequant_4bit_patched(weight, quant_state):
    """
    Patched dequantization function that properly handles QuantState objects
    """
    if hasattr(quant_state, 'scales'):
        # New bitsandbytes version with QuantState object
        scales = quant_state.scales
        zeros = quant_state.zeros if hasattr(quant_state, 'zeros') else None
        g_idx = quant_state.g_idx if hasattr(quant_state, 'g_idx') else None

        # Get original weight shape
        original_shape = weight.shape

        # Handle 4-bit quantization
        if weight.dtype == torch.uint8:
            # Reshape weight to match expected dimensions
            weight = weight.reshape(-1, weight.shape[-1])
            weight_deq = dequantize_4bit(weight, scales, zeros, g_idx)
            # Restore original shape
            weight_deq = weight_deq.reshape(original_shape)
        else:
            weight_deq = weight

        return weight_deq
    else:
        # Fallback for older versions or different quantization schemes
        return weight

def create_model_and_transforms():
    # ... existing code ...

    # Update the Linear4bit forward lambda with proper shape handling
    def linear_forward_wrapper(self, x):
        weight = dequant_4bit_patched(self.weight, self.weight.quant_state)
        return F.linear(x, weight, self.bias)

    # Patch the Linear4bit class
    for module in model.modules():
        if type(module).__name__ == 'Linear4bit':
            module.forward = types.MethodType(linear_forward_wrapper, module)

    # ... rest of existing code ...

# ---------------------
# Model Initialization
# ---------------------
def load_model():
    dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype
    )

    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        device_map="auto",
        torch_dtype=dtype
    )

    return model

# ---------------------
# LoRA Configuration
# ---------------------
def setup_lora(model):
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0,  # Keep dropout at 0 for better stability
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Freeze non-LoRA params
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad_(False)

    return model

# ----------------------
# Global Configuration
# ----------------------
max_seq_length = 1024  # Define this at the top level

# Compilation configuration
torch_compile_options = {
    "fullgraph": True,
    "dynamic": True,
    "options": {
        "epilogue_fusion": True,
        "max_autotune": True,
        "shape_padding": True,
        "triton.cudagraphs": True
    }
}

# For debugging and monitoring
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = False
torch._dynamo.config.cache_size_limit = 64  # Limit cache size to prevent memory issues

# Configure inductor optimizations through environment variables
os.environ["TORCH_INDUCTOR_OPTIMIZE_CUDAGRAPHS"] = "1"
os.environ["TORCH_INDUCTOR_MAX_AUTOTUNE"] = "1"
os.environ["TORCH_INDUCTOR_USE_TRITON"] = "1"
os.environ["TORCH_INDUCTOR_PERMUTE_FUSION"] = "1"
os.environ["TORCH_INDUCTOR_PATTERN_MATCHER"] = "1"
os.environ["TORCH_INDUCTOR_EPILOGUE_FUSION"] = "1"
os.environ["TORCH_INDUCTOR_COORDINATE_DESCENT"] = "1"

# Debug settings
os.environ["TORCH_LOGS"] = "+dynamo,+aot,+inductor"
os.environ["TORCHDYNAMO_REPORT_GUARD_FAILURES"] = "1"
os.environ["TORCHDYNAMO_COMPILE_VERBOSE"] = "1"
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

# Dynamo specific settings
torch._dynamo.config.dynamic_shapes = True
torch._dynamo.config.automatic_dynamic_shapes = True
torch._dynamo.config.optimize_ddp = True
torch._dynamo.config.assume_static_by_default = False

# ----------------------
# Compiled Model Components
# ----------------------
@torch.compile(**torch_compile_options)
def compiled_mlp(hidden_states, module=None):
    down_proj = module.down_proj(
        module.act_fn(module.gate_proj(hidden_states)) * module.up_proj(hidden_states)
    )
    return down_proj

@torch.compile(**torch_compile_options)
def compiled_rmsnorm(hidden_states, weight, eps=1e-6):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return hidden_states * weight

# Compile the forward pass of LlamaMLP
@torch.compile(**torch_compile_options)
def compiled_llama_mlp(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

# ----------------------
# Model Components Compilation
# ----------------------
@torch.compile(**torch_compile_options)
def compiled_attention_forward(hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, module=None):
    # Add dynamic sequence length handling
    bsz, q_len, _ = hidden_states.size()
    head_dim = module.head_dim
    num_heads = module.num_heads
    hidden_size = module.hidden_size
    scaling = 1.0 / math.sqrt(head_dim)

    # QKV projections
    query_states = module.q_proj(hidden_states) * scaling
    key_states = module.k_proj(hidden_states)
    value_states = module.v_proj(hidden_states)

    # Reshape for attention
    query_states = query_states.view(bsz, -1, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, -1, num_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, -1, num_heads, head_dim).transpose(1, 2)

    # Handle KV cache
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    if use_cache:
        past_key_value = (key_states, value_states)

    # Attention computation
    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_weights = attn_weights.to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    # Reshape output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)

    # Final projection
    attn_output = module.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return (attn_output, past_key_value, attn_weights) if output_attentions else (attn_output, past_key_value)

@torch.compile(**torch_compile_options)
def compiled_layernorm(hidden_states, weight, variance_epsilon=1e-6):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return weight * hidden_states

# ----------------------
# Model Patching
# ----------------------
def patch_model_for_compile(model):
    def create_attention_forward(module):
        def forward(hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False):
            return compiled_attention_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                module=module
            )
        return forward

    # Patch attention modules
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            module.forward = create_attention_forward(module)
            # Store original forward for reference
            module._original_forward = module.__class__.forward

    # Add LayerNorm patching
    for module in model.modules():
        if isinstance(module, LlamaRMSNorm):
            def create_forward(mod):
                def forward(x):
                    return compiled_layernorm(x, mod.weight, mod.variance_epsilon)
                return forward
            module.forward = create_forward(module)

    return model

# ----------------------
# Model Setup
# ----------------------
def setup_model():
    dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=dtype,
    )

    # Apply patches before LoRA
    model = patch_model_for_compile(model)

    # Setup LoRA
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing after compilation
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Freeze non-LoRA parameters
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad_(False)

    return model

# ------------------
# Training Setup
# ------------------
def setup_trainer(model, tokenizer):
    # Load dataset from URL
    dataset = load_dataset(
        "json",
        data_files={
            "train": "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
        },
        split="train[:10%]"
    )

    # Define data preprocessing function
    def preprocess_function(examples):
        # Create conversation pairs
        conversations = []
        for text in examples['text']:
            # Format each conversation as a list of messages
            messages = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": text}  # Using same text for demo
            ]
            conversations.append(messages)

        # Apply chat template
        formatted_texts = []
        for conv in conversations:
            try:
                formatted = tokenizer.apply_chat_template(
                    conv,
                    tokenize=False,
                    add_generation_prompt=False
                )
                formatted_texts.append(formatted)
            except Exception as e:
                print(f"Error formatting conversation: {e}")
                formatted_texts.append("")  # Add empty string as fallback

        # Tokenize with padding and truncation
        tokenized = tokenizer(
            formatted_texts,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors=None
        )

        return tokenized

    # Process the dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=100,  # Smaller batch size for processing
        num_proc=4,
        remove_columns=dataset.column_names,
        desc="Processing dataset",
    )

    # Configure training arguments
    training_args = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=1,
        max_steps=10,
        logging_steps=1,
        output_dir="outputs",
        seed=3407,
        max_seq_length=max_seq_length,
        fp16=model.get_input_embeddings().weight.dtype == torch.float16,
        bf16=model.get_input_embeddings().weight.dtype == torch.bfloat16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    return SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=None,
    )

# ------------------
# Monitoring Setup
# ------------------
def setup_monitoring():
    # Basic logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Setup dynamo logging
    dynamo_logging = logging.getLogger("torch._dynamo")
    dynamo_logging.setLevel(logging.INFO)

    # Setup inductor logging
    inductor_logging = logging.getLogger("torch._inductor")
    inductor_logging.setLevel(logging.INFO)

    # Add file handler for persistent logging
    fh = logging.FileHandler('compilation_log.txt')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add handlers
    dynamo_logging.addHandler(fh)
    inductor_logging.addHandler(fh)

def log_compilation_stats():
    """Log compilation statistics and performance metrics"""
    try:
        print("\nCompilation Statistics:")
        print("Graph breaks:", torch._dynamo.utils.graph_break_count())
        print("Compilation times:", torch._dynamo.utils.compile_times())

        # Log guard failures if any
        guard_failures = torch._dynamo.utils.guard_failures()
        if guard_failures:
            print("\nGuard Failures:")
            for failure in guard_failures:
                print(f"- {failure}")

        # Log optimization stats
        print("\nOptimization Statistics:")
        print("Graphs optimized:", torch._dynamo.utils.optimized_graph_count())
        print("Unique graphs:", torch._dynamo.utils.unique_graph_count())

    except Exception as e:
        print(f"Error collecting compilation stats: {str(e)}")

def log_memory_stats():
    """Log detailed memory statistics"""
    try:
        print("\nMemory Statistics:")
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

        # Log peak memory stats
        print("\nPeak Memory Usage:")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

        # Memory fragmentation
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        fragmentation = (reserved - allocated) / reserved if reserved > 0 else 0
        print(f"\nMemory Fragmentation: {fragmentation:.2%}")

    except Exception as e:
        print(f"Error collecting memory stats: {str(e)}")

def monitor_compilations():
    compilation_count = torch._dynamo.utils.compile_times().__len__()
    print(f"Total compilations: {compilation_count}")
    if compilation_count > 30:
        print("Warning: Excessive recompilations detected!")
    return compilation_count

# ------------
# Main Script
# ------------
if __name__ == "__main__":
    # Set up HuggingFace token
    os.environ["HF_TOKEN"] = "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"

    # Setup monitoring
    setup_monitoring()

    print("\n=== Starting Model Loading ===")
    print("Initial GPU Memory:")
    print(torch.cuda.memory_summary())

    print("\nLoading model...")
    model = load_model()
    print("Model loaded successfully!")

    print("\nGPU Memory after model load:")
    print(torch.cuda.memory_summary())

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        token=os.environ["HF_TOKEN"]
    )
    tokenizer.padding_side = "right"
    print("Tokenizer loaded successfully!")

    print("\nSetting up LoRA...")
    model = setup_lora(model)
    model.enable_input_require_grads()
    print("LoRA setup complete!")

    print("\nGPU Memory after LoRA setup:")
    print(torch.cuda.memory_summary())

    print("\nSetting up trainer...")
    trainer = setup_trainer(model, tokenizer)
    print("Trainer setup complete!")

    print("\nStarting compilation monitoring...")
    # Configure Dynamo for verbose output and debugging
    torch._dynamo.config.verbose = True
    torch._dynamo.config.suppress_errors = False

    # Configure inductor settings for debugging
    torch._inductor.config.debug = True
    torch._inductor.config.verbose_progress = True

    # Set environment variables for detailed logging
    os.environ["TORCH_LOGS"] = "+dynamo,+aot,+inductor"
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["TORCHDYNAMO_REPORT_GUARD_FAILURES"] = "1"

    print("\nStarting training...")
    try:
        train_result = trainer.train()
        print("\nTraining Results:")
        print(train_result)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\nFinal GPU Memory Usage:")
    print(torch.cuda.memory_summary())

    log_memory_stats()

    print("\nTraining completed!")

    # Add to your training loop
    compilation_count = monitor_compilations()

@torch.compile(**torch_compile_options)
def compiled_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
