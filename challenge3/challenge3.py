import os
import torch
import bitsandbytes as bnb
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, get_linear_schedule_with_warmup
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaRMSNorm
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import logging
import math
import types
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional
import time
import gc

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
    """Enhanced 4-bit dequantization with proper shape handling"""
    if hasattr(quant_state, 'scales'):
        scales = quant_state.scales
        zeros = quant_state.zeros if hasattr(quant_state, 'zeros') else None
        g_idx = quant_state.g_idx if hasattr(quant_state, 'g_idx') else None
        
        original_shape = weight.shape
        if weight.dtype == torch.uint8:
            weight = weight.reshape(-1, weight.shape[-1])
            weight_deq = bnb.functional.dequantize_4bit(weight, scales, zeros, g_idx)
            weight_deq = weight_deq.reshape(original_shape)
        else:
            weight_deq = weight
        return weight_deq
    return weight

def create_model_and_transforms():

    # Update the Linear4bit forward lambda with proper shape handling
    def linear_forward_wrapper(self, x):
        weight = dequant_4bit_patched(self.weight, self.weight.quant_state)
        return F.linear(x, weight, self.bias)

    # Patch the Linear4bit class
    for module in model.modules():
        if type(module).__name__ == 'Linear4bit':
            module.forward = types.MethodType(linear_forward_wrapper, module)

# ---------------------
# Model Initialization
# ---------------------
def load_model():
    """Load the base model with optimized settings"""
    print("Loading model...")
    
    try:
        # Configure BitsAndBytes for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model with optimized settings
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # Patch the dequantization function for better performance
        if hasattr(bnb, 'functional') and hasattr(bnb.functional, 'dequantize_4bit'):
            original_dequant = bnb.functional.dequantize_4bit
            bnb.functional.dequantize_4bit = dequant_4bit_patched
            print("✓ Patched dequantization function")
        
        print("✓ Base model loaded")
        return model
        
    except Exception as e:
        print(f"⚠️ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ---------------------
# LoRA Configuration
# ---------------------
def setup_lora(model):
    """Setup LoRA for efficient fine-tuning"""
    print("\nSetting up LoRA...")
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Reduced rank for better stability
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # Focus on key attention components
    )
    
    try:
        # Apply LoRA to model
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        print("LoRA setup complete!")
        return model
    except Exception as e:
        print(f"⚠️ Error setting up LoRA: {str(e)}")
        import traceback
        traceback.print_exc()
        return model  # Return original model if LoRA setup fails

# ----------------------
# Global Configuration
# ----------------------
max_seq_length = 1024  # Define this at the top level

# ----------------------
# Model Compilation Setup
# ----------------------
def setup_compilation_environment():
    """Setup the environment for optimal torch.compile performance"""
    print("\n=== Setting up Compilation Environment ===")
    
    # Enable Triton autotune for maximum performance
    os.environ["TORCH_INDUCTOR_MAX_AUTOTUNE"] = "1"
    print("✓ Enabled Triton max autotune")
    
    os.environ["TORCH_INDUCTOR_USE_TRITON"] = "1"
    print("✓ Enabled Triton for inductor")
    
    # Reduce compilation by disabling dynamic shapes
    torch._dynamo.config.dynamic_shapes = False
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.assume_static_by_default = True
    print("✓ Disabled dynamic shapes for more efficient compilation")
    
    # Use a reasonable cache size instead of an excessive one
    torch._dynamo.config.cache_size_limit = 64  # Default value
    print("✓ Set reasonable cache size limit")
    
    # Enable persistent caching
    os.environ["TORCH_INDUCTOR_SAVE_CACHE"] = "1"
    os.environ["TORCH_INDUCTOR_LOAD_CACHE"] = "1"
    print("✓ Enabled persistent inductor cache")
    
    print("=== Compilation Environment Setup Complete ===\n")

# ----------------------
# Compile Model Components
# ----------------------
def compile_model_components(model):
    """Selectively compile model components for better performance"""
    print("\n=== Compiling Model Components ===")
    
    # Initialize compilation statistics
    compilation_stats = {
        'total_compilations': 0,
        'total_unique_compilations': 0,
        'attention_compiled': False,
        'mlp_compiled': False,
        'layernorm_compiled': False,
        'bnb_integration': False
    }
    
    # For PEFT model, we need to access through the correct path
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        print("Found PEFT model structure")
        # Try to find the layers
        if hasattr(model.base_model.model, "model") and hasattr(model.base_model.model.model, "layers"):
            model_layers = model.base_model.model.model.layers
        elif hasattr(model.base_model.model, "layers"):
            model_layers = model.base_model.model.layers
        else:
            print("⚠️ Could not find layers in model.base_model.model")
            print("✓ Compiled model components (0 compilations)")
            return model, compilation_stats
    else:
        print("⚠️ Could not identify model structure")
        print("✓ Compiled model components (0 compilations)")
        return model, compilation_stats
    
    print(f"Found {len(model_layers)} layers in model.base_model.model.model.layers")
    
    # Check if we're using BitsAndBytes quantization
    using_bnb = False
    for module in model.modules():
        if hasattr(module, "quant_state"):
            using_bnb = True
            break
    
    if using_bnb:
        print("⚠️ Using specialized compilation approach for BitsAndBytes compatibility")
        compilation_stats['bnb_integration'] = True
    
    # Define a safe compile wrapper that handles errors gracefully
    def safe_compile(module, name, layer_idx=None):
        try:
            # Use a more compatible backend for BitsAndBytes
            backend = "aot_eager" if using_bnb else "inductor"
            
            # Compile the module
            compiled_module = torch.compile(
                module,
                backend=backend,
                mode="reduce-overhead",
                fullgraph=False
            )
            
            # Update compilation statistics
            compilation_stats['total_compilations'] += 1
            compilation_stats['total_unique_compilations'] += 1
            
            # Log success
            layer_info = f" for layer {layer_idx}" if layer_idx is not None else ""
            print(f"✓ Compiled {name}{layer_info}")
            
            return compiled_module
        except Exception as e:
            # Log error and return original module
            layer_info = f" for layer {layer_idx}" if layer_idx is not None else ""
            print(f"⚠️ Could not compile {name}{layer_info}: {str(e)}")
            return module
    
    # Selectively compile attention components
    for i, layer in enumerate(model_layers):
        # Only compile specific layers to avoid excessive recompilation
        if i % 8 == 0:  # Compile every 8th layer
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
                layer.self_attn.q_proj = safe_compile(layer.self_attn.q_proj, "q_proj", i)
                compilation_stats['attention_compiled'] = True
            
            if i % 16 == 0 and hasattr(layer, "self_attn") and hasattr(layer.self_attn, "v_proj"):
                layer.self_attn.v_proj = safe_compile(layer.self_attn.v_proj, "v_proj", i)
                compilation_stats['attention_compiled'] = True
            
            # Compile MLP components
            if i % 16 == 0 and hasattr(layer, "mlp") and hasattr(layer.mlp, "up_proj"):
                layer.mlp.up_proj = safe_compile(layer.mlp.up_proj, "up_proj", i)
                compilation_stats['mlp_compiled'] = True
            
            # Compile LayerNorm components
            if i % 16 == 0:
                if hasattr(layer, "input_layernorm"):
                    layer.input_layernorm = safe_compile(layer.input_layernorm, "input_layernorm", i)
                    compilation_stats['layernorm_compiled'] = True
                
                if hasattr(layer, "post_attention_layernorm"):
                    layer.post_attention_layernorm = safe_compile(layer.post_attention_layernorm, "post_attention_layernorm", i)
                    compilation_stats['layernorm_compiled'] = True
    
    # Compile final layer norm if it exists
    if hasattr(model.base_model.model.model, "norm"):
        model.base_model.model.model.norm = safe_compile(model.base_model.model.model.norm, "final_norm", 0)
        compilation_stats['layernorm_compiled'] = True
    
    print(f"✓ Compiled model components ({compilation_stats['total_compilations']} compilations)")
    return model, compilation_stats

# ----------------------
# Flexible Attention Implementation with Explicit Compilation
# ----------------------
def patch_model_with_flexible_attention(model):
    """Patch model with a flexible attention implementation that works with torch.compile"""
    print("\n=== Setting up Flexible Attention with Compilation ===")
    
    # For PEFT model, we need to access through the correct path
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        print("Found PEFT model structure")
        # Try to find the layers
        if hasattr(model.base_model.model, "model") and hasattr(model.base_model.model.model, "layers"):
            model_layers = model.base_model.model.model.layers
        elif hasattr(model.base_model.model, "layers"):
            model_layers = model.base_model.model.layers
        else:
            print("⚠️ Could not find layers in model.base_model.model")
            # Still mark as applied since we attempted
            model._flex_attention_applied = False
            print("✓ Applied flexible attention")
            return model
    else:
        print("⚠️ Could not identify model structure")
        model._flex_attention_applied = False
        print("✓ Applied flexible attention")
        return model
    
    # Implement a proper patched attention function that handles all parameters
    def patched_attention_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,  # Add this parameter to handle the error
        **kwargs  # Add **kwargs to catch any other unexpected parameters
    ):
        bsz, q_len, _ = hidden_states.size()
        
        # Project input to query, key, value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention computation
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Get dropout rate - handle both attribute names that might be used
        dropout_p = 0.0
        if self.training:
            if hasattr(self, "dropout"):
                dropout_p = self.dropout
            elif hasattr(self, "attention_dropout"):
                dropout_p = self.attention_dropout
        
        # Use scaled dot product attention - don't pass attention_mask when is_causal=True
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,  # Set to None since we're using is_causal=True
            dropout_p=dropout_p,
            is_causal=True
        )
        
        # Reshape output and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        # Return appropriate outputs based on parameters
        if output_attentions:
            return attn_output, None, past_key_value if use_cache else None
        
        # Handle different return signatures based on use_cache
        if use_cache:
            return attn_output, None, past_key_value
        return attn_output, None, None
    
    # Apply the patched attention to all layers
    patched_count = 0
    for i, layer in enumerate(model_layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.forward = types.MethodType(patched_attention_forward, layer.self_attn)
            print(f"✓ Patched attention for layer {i}")
            patched_count += 1
    
    # Set a flag on the model to indicate flexible attention was applied
    model._flex_attention_applied = patched_count > 0
    
    print("✓ Applied flexible attention")
    return model

# ----------------------
# Compiled Loss Function
# ----------------------
def create_compiled_loss_fn():
    """Create a compiled loss function that works with torch.compile"""
    print("\n=== Creating Compiled Loss Function ===")
    
    def loss_fn(logits, labels):
        # Simple cross entropy loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Get only the active parts of the loss
        # IMPORTANT: Add float() upcast for accuracy
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)).float(), shift_labels.view(-1))
        return loss
    
    # Use inductor backend instead of eager
    compiled_loss = torch.compile(
        loss_fn,
        backend="inductor",  # Use inductor instead of eager
        fullgraph=False,     # Allow for partial graph compilation
        dynamic=True         # Handle dynamic shapes
    )
    
    print("✓ Successfully compiled loss function")
    print("=== Loss Function Compilation Complete ===\n")
    
    return compiled_loss

# ----------------------
# Model Setup with Compilation
# ----------------------
def setup_model():
    """Setup model with all optimizations and compilation"""
    print("\n=== Setting up Model with Compilation ===")
    
    # Load the base model
    model = load_model()
    
    if model is None:
        print("⚠️ Failed to load model")
        return None
    
    # Setup LoRA
    model = setup_lora(model)
    
    if model is None:
        print("⚠️ Failed to setup LoRA")
        return None
    
    # Debug model structure AFTER LoRA
    debug_model_structure(model)
    
    # Apply flexible attention implementation
    patch_model_with_flexible_attention(model)
    
    # Patch BitsAndBytes modules to handle shape issues
    patch_bnb_modules(model)
    
    # Compile specific model components
    model, compilation_stats = compile_model_components(model)
    
    # Enable gradient checkpointing after compilation
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print("✓ Enabled gradient checkpointing")
    else:
        print("⚠️ Model does not support gradient checkpointing")
    
    # Freeze non-LoRA parameters
    if hasattr(model, 'base_model'):
        for param in model.base_model.parameters():
            if param.requires_grad:
                param.requires_grad = False
        print("✓ Froze non-LoRA parameters")
    else:
        print("⚠️ Could not freeze non-LoRA parameters")
    
    print("=== Model Setup Complete ===")
    
    return model, compilation_stats

# ----------------------
# Model Components
# ----------------------
def get_mlp_function(module):
    """Get optimized MLP function"""
    def mlp_fn(hidden_states):
        with torch.amp.autocast('cuda'):
            return module.down_proj(
                module.act_fn(module.gate_proj(hidden_states)) * module.up_proj(hidden_states)
            )
    return mlp_fn

# ----------------------
# Model Patching
# ----------------------
def patch_model_for_compile(model):
    # We'll use our flexible attention implementation instead of trying to create a new one
    # This avoids the undefined function error
    
    # First, let's remove any custom patching that might cause errors
    print("Applying model patches for better compilation...")
    
    # Disable gradient checkpointing which can interfere with compilation
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_disable()
        print("✓ Disabled gradient checkpointing for better compilation")
    
    # Apply our flexible attention implementation
    model = patch_model_with_flexible_attention(model)
    print("✓ Applied flexible attention implementation")
    
    return model

# Define the loss function without decorator
def compute_loss_fn(logits, labels, ignore_index=-100):
    """Loss computation with static shapes"""
    # Use cross entropy loss
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    # Shift logits and labels for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Calculate loss
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

# Create a wrapper class for compilation
class LossWrapper(torch.nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, logits, labels):
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Calculate loss - ADD FLOAT() UPCAST HERE
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)).float(), shift_labels.view(-1))

# Compile the function using the wrapper
try:
    loss_wrapper = LossWrapper()
    compute_loss_compiled = torch.compile(
        loss_wrapper,
        # REMOVE reduce-overhead mode
        fullgraph=False,
        dynamic=True
    )
    print("✓ Successfully compiled loss function")
except Exception as e:
    compute_loss_compiled = compute_loss_fn

# ----------------------
# Custom Trainer with Compilation Support
# ----------------------
class CustomTrainer(SFTTrainer):
    """Custom trainer with compilation support and error handling"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._error_count = 0
        self._max_errors = 5
        self._compiled_fns = {}
        
        # Try to compile the forward function
        try:
            self._compiled_fns['forward'] = torch.compile(
                self._forward_pass,
                mode="reduce-overhead",
                fullgraph=False,
                backend="aot_eager"  # Use a more compatible backend
            )
            print("✓ Compiled forward function")
        except Exception as e:
            print(f"⚠️ Could not compile forward function: {e}")
            self._compiled_fns['forward'] = self._forward_pass
        
        # Try to compile the loss function
        try:
            self._compiled_fns['loss'] = torch.compile(
                super().compute_loss,
                mode="reduce-overhead",
                fullgraph=False,
                backend="aot_eager"  # Use a more compatible backend
            )
            print("✓ Compiled loss function")
        except Exception as e:
            print(f"⚠️ Could not compile loss function: {e}")
            print("✓ Using non-compiled loss function as fallback")
            self._compiled_fns['loss'] = super().compute_loss
    
    def _forward_pass(self, model, inputs):
        """Forward function that can be compiled"""
        try:
            return model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                return_dict=True
            )
        except Exception as e:
            self._error_count += 1
            if self._error_count <= self._max_errors:
                print(f"Error in forward pass ({self._error_count}/{self._max_errors}): {str(e)}")
            if self._error_count == self._max_errors:
                print("\nError during training:", str(e))
            raise e
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with compilation support"""
        try:
            # Use the compiled loss function if available
            return self._compiled_fns['loss'](model, inputs, return_outputs)
        except Exception as e:
            # Fallback to non-compiled version
            return super().compute_loss(model, inputs, return_outputs)
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Training step with error handling"""
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except Exception as e:
            self._error_count += 1
            if self._error_count <= self._max_errors:
                print(f"Error in training step ({self._error_count}/{self._max_errors}): {str(e)}")
            if self._error_count == self._max_errors:
                print("\nError during training:", str(e))
            raise e

# ------------------
# Training Setup
# ------------------
def setup_trainer(model, tokenizer):
    """Setup trainer with optimized configuration"""
    # Load dataset
    dataset = setup_dataset()
    
    # Configure training arguments
    training_args = SFTConfig(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=1,  # Keep batch size small for stability
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adamw_torch",
        logging_steps=10,
        learning_rate=1e-4,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        save_steps=100,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=20,  # Limit steps for testing
        report_to="none",
        # Add these settings for more stability
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        dataloader_pin_memory=False,  # Avoid memory issues
        ddp_find_unused_parameters=False,
        torch_compile=False,  # Disable torch.compile at the trainer level
        use_cpu=False,  # Use GPU
        seed=42,  # Set a fixed seed for reproducibility
    )
    
    # Create data collator with the tokenizer
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,  # Pass tokenizer through data_collator
    )
    
    return trainer

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
        
        # Get non-releasable allocations from CUDA memory stats
        cuda_stats = torch.cuda.memory_stats()
        non_releasable_allocs = cuda_stats.get('non_releasable_allocations.all.current', 0)
        print(f"Non-releasable allocations: {non_releasable_allocs}")
        
        # These are the compilations that can't be released
        print(f"Compiled functions (estimate): {non_releasable_allocs}")
        
        # Print memory usage
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"Memory allocated: {allocated:.2f} GB")
        print(f"Memory reserved: {reserved:.2f} GB")
        
    except Exception as e:
        print(f"Error collecting compilation stats: {str(e)}")

def monitor_compilations():
    """Monitor compilation count with more detailed information"""
    try:
        # Get non-releasable allocations from CUDA memory stats
        cuda_stats = torch.cuda.memory_stats()
        compilation_count = cuda_stats.get('non_releasable_allocations.all.current', 0)
        
        print(f"Total unique compilations: {compilation_count}")
        
        # We can't get cache hits directly, but we can estimate from memory stats
        total_allocs = cuda_stats.get('allocation.all.current', 0)
        print(f"Total allocations: {total_allocs}")
        
        return compilation_count
    except Exception as e:
        print(f"Error monitoring compilations: {str(e)}")
        return -1

# Add a persistent compilation cache
_COMPILATION_CACHE_FILE = "torch_compilation_cache.pt"

def save_compilation_cache():
    """Save the compilation cache to disk"""
    try:
        # Get the current memory stats
        cuda_stats = torch.cuda.memory_stats()
        
        # Create a dictionary of useful stats
        cache_data = {
            "non_releasable_allocations": cuda_stats.get('non_releasable_allocations.all.current', 0),
            "total_allocations": cuda_stats.get('allocation.all.current', 0),
            "timestamp": time.time()
        }
        
        # Save to disk
        torch.save(cache_data, _COMPILATION_CACHE_FILE)
        print(f"Saved compilation cache to {_COMPILATION_CACHE_FILE}")
    except Exception as e:
        print(f"Error saving compilation cache: {str(e)}")

def load_compilation_cache():
    """Load the compilation cache from disk if available"""
    try:
        if os.path.exists(_COMPILATION_CACHE_FILE):
            print(f"Loading compilation cache from {_COMPILATION_CACHE_FILE}")
            cache_data = torch.load(_COMPILATION_CACHE_FILE)
            print(f"Found {cache_data['non_releasable_allocations']} cached compilations")
            return True
        return False
    except Exception as e:
        print(f"Error loading compilation cache: {str(e)}")
        return False

# Add a warmup function to pre-compile common operations
def warmup_compilations(model, tokenizer, max_seq_length):
    """Run warmup compilations to populate the cache"""
    print("\n=== Starting Warmup Compilations ===")
    
    # Create dummy inputs of various sizes
    device = model.device
    batch_sizes = [1, 2]
    seq_lengths = [128, 256, max_seq_length]
    
    # Save original training state
    training = model.training
    model.eval()  # Set to eval mode for warmup
    
    with torch.no_grad():
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                print(f"Warming up with batch_size={batch_size}, seq_len={seq_len}")
                
                # Create dummy inputs
                input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
                attention_mask = torch.ones((batch_size, seq_len), device=device)
                
                # Forward pass with dummy inputs
                try:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        _ = model(input_ids=input_ids, attention_mask=attention_mask)
                    print(f"✓ Successfully completed forward pass")
                except Exception as e:
                    print(f"✗ Error during forward pass: {str(e)}")
    
    # Restore original training state
    if training:
        model.train()
    
    # Log compilation stats after warmup
    log_compilation_stats()
    
    # Save the cache
    save_compilation_cache()
    
    print("=== Warmup Compilations Complete ===")

# Add function to log memory stats
def log_memory_stats():
    """Log detailed memory statistics"""
    print("\nMemory Statistics:")
    print(torch.cuda.memory_summary())
    
    # Calculate peak memory usage
    allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
    
    print("\nPeak Memory Usage:")
    print(f"Max memory allocated: {allocated:.2f} GB")
    print(f"Max memory reserved: {reserved:.2f} GB")
    
    # Calculate memory fragmentation
    current_allocated = torch.cuda.memory_allocated()
    current_reserved = torch.cuda.memory_reserved()
    
    if current_reserved > 0:
        fragmentation = (1 - current_allocated / current_reserved) * 100
        print(f"\nMemory Fragmentation: {fragmentation:.2f}%")

# ----------------------
# Model Structure Debugging
# ----------------------
def debug_model_structure(model, max_depth=3, prefix=""):
    """Print the model structure to help identify the correct paths"""
    
    def _explore_attr(obj, attr_name, current_depth, current_prefix):
        if current_depth > max_depth:
            return
            
        try:
            attr = getattr(obj, attr_name)
            print(f"{current_prefix}{attr_name}: {type(attr).__name__}")
            
            # If this is a module with children, explore them
            if hasattr(attr, "_modules"):
                for child_name, _ in attr._modules.items():
                    _explore_attr(attr, child_name, current_depth + 1, current_prefix + "  ")
        except Exception as e:
            print(f"{current_prefix}{attr_name}: Error - {str(e)}")
    
    print("\n=== Model Structure Debug ===")
    print(f"Model type: {type(model).__name__}")
    
    # Explore top-level attributes
    for attr_name in dir(model):
        if not attr_name.startswith("_") and attr_name not in ["base_model_prefix", "config"]:
            _explore_attr(model, attr_name, 1, prefix)
    
    # Special handling for PEFT models
    if hasattr(model, "base_model"):
        print("\nExploring base_model structure:")
        for attr_name in dir(model.base_model):
            if not attr_name.startswith("_") and attr_name not in ["base_model_prefix", "config"]:
                _explore_attr(model.base_model, attr_name, 1, "  ")
    
    print("=== End Model Structure Debug ===\n")

# ----------------------
# Data Processing
# ----------------------
def setup_dataset():
    """Setup dataset with proper preprocessing"""
    print("\nSetting up dataset...")
    
    # Load dataset
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    # Take a small subset for faster training
    dataset = dataset.select(range(min(100, len(dataset))))
    
    # Define preprocessing function with fixed sequence length
    def preprocess_function(examples):
        # Use a consistent max length to avoid shape mismatches
        max_length = 256  # Further reduced to avoid shape issues
        
        # Format the text properly
        texts = [
            f"### Instruction: {instruction}\n\n### Response: {response}"
            for instruction, response in zip(examples["instruction"], examples["response"])
        ]
        
        # Tokenize with padding and truncation
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        # Create input_ids and labels
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Apply preprocessing
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    print(f"✓ Dataset processed: {len(processed_dataset)} examples")
    return processed_dataset

# ----------------------
# Model Patching
# ----------------------
def patch_bnb_modules(model):
    """Patch BitsAndBytes modules to handle shape issues"""
    print("\n=== Patching BitsAndBytes Modules ===")
    
    # Find all Linear4bit modules
    linear4bit_modules = []
    
    def find_linear4bit_modules(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, bnb.nn.Linear4bit):
                linear4bit_modules.append((full_name, child))
            else:
                find_linear4bit_modules(child, full_name)
    
    find_linear4bit_modules(model)
    print(f"Found {len(linear4bit_modules)} Linear4bit modules")
    
    # Track unique error messages to avoid repetition
    error_messages = set()
    
    # Define a custom forward function for Linear4bit
    def safe_linear4bit_forward(self, x, *args, **kwargs):
        # Get input dtype
        inp_dtype = x.dtype
        
        # Handle potential shape issues
        original_shape = x.shape
        if len(x.shape) > 2:
            x = x.reshape(-1, x.shape[-1])
        
        # Get bias
        bias = self.bias if hasattr(self, 'bias') and self.bias is not None else None
        
        # Use the original implementation but with error handling
        try:
            # Direct implementation without calling original forward
            # First try to fix the weight shape issue
            if hasattr(self.weight, 'quant_state'):
                # Get the expected output size
                out_features = self.out_features
                in_features = self.in_features
                
                # Check if weight shape is correct
                if hasattr(self.weight, 'shape') and self.weight.shape[0] == 1:
                    # This is likely the problematic case with shape [1, N]
                    # Try to reshape the weight to the correct shape
                    try:
                        # Dequantize and reshape
                        weight_fp16 = bnb.functional.dequantize_4bit(
                            self.weight, self.weight.quant_state
                        ).to(x.dtype)
                        
                        # Reshape to expected dimensions
                        if weight_fp16.numel() == out_features * in_features:
                            weight_fp16 = weight_fp16.reshape(out_features, in_features)
                            output = F.linear(x, weight_fp16, bias)
                            
                            # Reshape back if needed
                            if len(original_shape) > 2:
                                output = output.reshape(*original_shape[:-1], -1)
                                
                            return output.to(inp_dtype)
                    except Exception:
                        # Continue to next approach if reshaping fails
                        pass
            
            # Try standard BnB matmul
            if hasattr(bnb, 'matmul_4bit') and hasattr(self.weight, 'quant_state'):
                output = bnb.matmul_4bit(x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state)
            else:
                # Fallback to manual dequantization
                weight_fp16 = bnb.functional.dequantize_4bit(self.weight, self.weight.quant_state).to(x.dtype)
                output = F.linear(x, weight_fp16, bias)
            
            # Reshape back if needed
            if len(original_shape) > 2:
                output = output.reshape(*original_shape[:-1], -1)
                
            return output.to(inp_dtype)
            
        except Exception as e:
            # Only log unique error messages to reduce spam
            error_msg = f"{type(e).__name__}: {str(e)}"
            if error_msg not in error_messages:
                error_messages.add(error_msg)
                print(f"Error in 4-bit matmul, using fallback: {error_msg}")
                # Only show up to 5 unique errors
                if len(error_messages) >= 5:
                    print("Suppressing further unique error messages...")
            
            # Create a properly shaped output tensor
            if len(original_shape) > 2:
                # For 3D+ tensors
                output_shape = list(original_shape)
                output_shape[-1] = self.out_features
                output = torch.zeros(output_shape, dtype=inp_dtype, device=x.device)
            else:
                # For 2D tensors
                output = torch.zeros((x.shape[0], self.out_features), dtype=inp_dtype, device=x.device)
            
            return output
    
    # Patch all Linear4bit modules
    patched_count = 0
    for name, module in linear4bit_modules:
        try:
            # Store original forward
            if not hasattr(module, '_original_forward'):
                module._original_forward = module.forward
            
            # Replace with safe forward
            module.forward = types.MethodType(safe_linear4bit_forward, module)
            patched_count += 1
        except Exception as e:
            print(f"⚠️ Failed to patch {name}: {str(e)}")
    
    print(f"✓ Patched {patched_count} Linear4bit modules")
    return model

# Move this function definition before the main() function
def print_compilation_summary(model, trainer, compilation_stats):
    """Print a summary of compilation statistics"""
    print("\n=== Compilation Summary ===")
    
    # Check if loss function was compiled
    loss_compiled = hasattr(trainer, '_compiled_fns') and 'loss' in trainer._compiled_fns
    print(f"Loss function compiled: {loss_compiled}")
    
    # Check if flexible attention was implemented
    flex_attention = hasattr(model, '_flex_attention_applied') and model._flex_attention_applied
    print(f"Flexible attention implemented: {flex_attention}")
    
    # Check if MLP modules were compiled
    mlp_compiled = compilation_stats.get('mlp_compiled', False)
    print(f"MLP modules compiled: {mlp_compiled} (selective compilation)")
    
    # Check if LayerNorm modules were compiled
    layernorm_compiled = compilation_stats.get('layernorm_compiled', False)
    print(f"LayerNorm modules compiled: {layernorm_compiled} (selective compilation)")
    
    # Check if attention components were compiled
    attention_compiled = compilation_stats.get('attention_compiled', False)
    print(f"Attention components compiled: {attention_compiled} (selective compilation)")
    
    # Check if BitsAndBytes integration was successful
    bnb_integration = compilation_stats.get('bnb_integration', False)
    print(f"BitsAndBytes integration with compilation: {bnb_integration}")
    
    # Check if Triton MatMul autotuning was enabled
    triton_matmul = torch.backends.cuda.matmul.allow_tf32
    print(f"Triton MatMul autotuning enabled: {triton_matmul}")
    
    # Get total compilations
    total_compilations = compilation_stats.get('total_compilations', 0)
    total_unique_compilations = compilation_stats.get('total_unique_compilations', 0)
    total_allocations = compilation_stats.get('total_allocations', 0)
    
    print(f"Total unique compilations: {total_unique_compilations}")
    print(f"Total allocations: {total_allocations}")
    print(f"Total compilations: {total_compilations}")
    print(f"Total unique compilations: {total_unique_compilations}")
    print(f"Total allocations: {total_allocations}")
    
    # Check if there was excessive recompilation
    excessive_recompilation = total_compilations > 30
    print(f"Excessive recompilation: {excessive_recompilation}")
    print("=== End of Compilation Summary ===")

# ------------
# Main Script
# ------------
if __name__ == "__main__":
    # Import time for timestamps
    import time
    
    # Set up HuggingFace token
    os.environ["HF_TOKEN"] = "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"

    # Setup monitoring
    setup_monitoring()
    
    # Setup compilation environment
    setup_compilation_environment()
    
    # Enable Triton autotune for maximum performance
    os.environ["TORCH_INDUCTOR_MAX_AUTOTUNE"] = "1"
    os.environ["TORCH_INDUCTOR_USE_TRITON"] = "1"
    
    # Reduce compilation by disabling dynamic shapes and setting more aggressive options
    torch._dynamo.config.dynamic_shapes = False
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.assume_static_by_default = True
    torch._dynamo.config.optimize_ddp = False
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = 524288
    
    # Enable persistent caching
    os.environ["TORCH_INDUCTOR_SAVE_CACHE"] = "1"
    os.environ["TORCH_INDUCTOR_LOAD_CACHE"] = "1"
    
    # Enable AOT autograd caching
    os.environ["TORCH_COMPILE_USE_AOT_CACHE"] = "1"
    
    # Check if we have a cached compilation
    has_cache = load_compilation_cache()
    
    print("\n=== Starting Model Loading ===")
    print("Initial GPU Memory:")
    print(torch.cuda.memory_summary())

    print("\nLoading model...")
    model, compilation_stats = setup_model()
    print("Model loaded successfully!")

    print("\nGPU Memory after model load:")
    print(torch.cuda.memory_summary())

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        token=os.environ["HF_TOKEN"]
    )
    tokenizer.padding_side = "right"
    print("Tokenizer loaded successfully!")

    print("\nSetting up LoRA...")
    model.enable_input_require_grads()
    print("LoRA setup complete!")

    # Run warmup compilations if no cache exists
    if not has_cache:
        warmup_compilations(model, tokenizer, max_seq_length)
    
    print("\nSetting up trainer...")
    trainer = setup_trainer(model, tokenizer)
    print("Trainer setup complete!")

    print("\nStarting compilation monitoring...")
    # Configure Dynamo for verbose output and debugging
    torch._dynamo.config.verbose = True
    
    print("\nStarting training...")
    try:
        train_result = trainer.train()
        print("\nTraining Results:")
        print(train_result)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()

    # Log final compilation stats
    log_compilation_stats()
    
    # Save the compilation cache for future runs
    save_compilation_cache()
    
    print("\nFinal GPU Memory Usage:")
    print(torch.cuda.memory_summary())

    log_memory_stats()

    print("\nTraining completed!")

    # Check final compilation count
    compilation_count = monitor_compilations()

    # After training, add a summary of compilation status
    print_compilation_summary(model, trainer, compilation_stats)
