import os
import sys
import subprocess
from packaging import version

# Check if bitsandbytes is installed and up-to-date.
try:
    import bitsandbytes
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes"])
    import bitsandbytes

if version.parse(bitsandbytes.__version__) < version.parse("0.39.0"):
    print("Updating bitsandbytes to the latest version...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "bitsandbytes"])
    print("Please restart the runtime after upgrading bitsandbytes and then re-run this code.")
    sys.exit(0)

# Set environment variables for optimal performance
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Explicitly set for 2 GPUs (T4 on Kaggle)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True,"
    "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
)

# Enable torch inductor for better performance (if Triton is installed)
os.environ["TORCH_COMPILE_BACKEND"] = "inductor"

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    CPUOffload,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from datasets import load_dataset
from torch.cuda.amp import GradScaler
import time

# --- Distributed Process Group Initialization for Kaggle T4 GPUs ---
def init_distributed():
    """Initialize process group for distributed training."""
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            rank=rank,
            world_size=world_size
        )
    
    return local_rank, world_size, rank

local_rank, world_size, rank = init_distributed()
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# --- Model & Quantization Setup ---
model_name = "unsloth/meta-Llama-3.1-8B-Instruct-bnb-4bit"
max_seq_length = 1024  # Adjusted for T4 memory constraints
batch_size = 1  
gradient_accumulation_steps = 4

# Improved BnB config for better performance
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Check for flash_attn availability
try:
    import flash_attn  # Only needed to check if available
    attn_impl = "flash_attention_2"
except ImportError:
    print("flash_attn package not installed. Using default attention implementation.")
    attn_impl = None

# Prepare kwargs for model loading. Only pass attn_implementation if flash_attn is available.
model_kwargs = {
    "device_map": {"": local_rank},
    "quantization_config": bnb_config,
}
if attn_impl is not None:
    model_kwargs["attn_implementation"] = attn_impl

# Load model with BnB quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    **model_kwargs
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"

# Disable caching for gradient checkpointing
model.config.use_cache = False

# --- Apply LoRA via PEFT ---
from peft import get_peft_model, LoraConfig, TaskType

# Optimized LoRA config with parameters that work well with FSDP
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# Set parameters for gradient requirements: only LoRA parameters are trainable.
with torch.no_grad():
    for name, param in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# --- Configure FSDP with optimized settings ---
# Use auto_wrap_policy to target transformer layers.
auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},
)

mixed_precision_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
    cast_forward_inputs=True,
)

cpu_offload = CPUOffload(offload_params=True)

# Build a list of modules to ignore for FSDP wrapping.
# Here we ignore modules that contain only frozen parameters (e.g. quantized parameters).
ignored_modules = []
for module in model.modules():
    params = list(module.parameters(recurse=False))
    if params and all(not p.requires_grad for p in params):
        ignored_modules.append(module)

# Configure FSDP.
# Note: When WORLD_SIZE is 1, FSDP may switch to NO_SHARD.
model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mixed_precision_policy,
    device_id=local_rank,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    limit_all_gathers=True,
    use_orig_params=True,
    cpu_offload=cpu_offload,
    ignored_modules=ignored_modules,  # Exclude frozen submodules from wrapping.
)

# Apply activation checkpointing to save memory.
non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)
check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=non_reentrant_wrapper,
    check_fn=check_fn,
)

# --- Training setup ---
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scaler = GradScaler()  # Note: FutureWarning regarding GradScaler usage

# --- TorchDynamo Config & Conditional torch.compile ---
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Check if Triton is installed.
try:
    import triton
    triton_installed = True
except ImportError:
    triton_installed = False

if hasattr(torch, 'compile') and triton_installed:
    try:
        print("Applying torch.compile to the model...")
        model = torch.compile(model, backend="inductor")
    except Exception as e:
        print(f"torch.compile failed: {e}. Falling back to eager mode.")
else:
    print("Skipping torch.compile (either not available or Triton is missing).")

# --- Dataset Preparation ---
url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
dataset = load_dataset("json", data_files={"train": url}, split="train[:5%]")  # Smaller sample for T4 GPUs

def process(examples):
    texts = examples["text"]
    tokenized = tokenizer(
        texts,
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return tokenized

dataset = dataset.map(process, batched=True, remove_columns=dataset.column_names)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

if world_size > 1:
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
else:
    sampler = None

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=2,  # Reduced for T4 GPUs
    pin_memory=True,
)

# --- Training Loop ---
model.train()
total_loss = 0.0
start_time = time.time()

print(f"Starting training on {world_size} GPU(s) - Designed for 2x Tesla T4 on Kaggle")

for step, batch in enumerate(dataloader):
    if step >= 10 * gradient_accumulation_steps:
        break
    
    inputs = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=inputs)
        loss = outputs.loss / gradient_accumulation_steps
    
    scaler.scale(loss).backward()
    total_loss += loss.item()
    
    if (step + 1) % gradient_accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        if rank == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            steps_per_sec = (step + 1) / elapsed
            print(f"Step {(step+1)//gradient_accumulation_steps} | Loss: {total_loss:.4f} | Steps/sec: {steps_per_sec:.4f}")
        
        total_loss = 0.0

if rank == 0:
    model.save_pretrained("./llama-3.1-8b-lora")
    print("Training complete. Model saved to ./llama-3.1-8b-lora")

if world_size > 1:
    dist.destroy_process_group()
