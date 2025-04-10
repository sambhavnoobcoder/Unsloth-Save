import os
import sys
import subprocess
from packaging import version

# Clear environment variables
if "MASTER_ADDR" in os.environ: del os.environ["MASTER_ADDR"]
if "MASTER_PORT" in os.environ: del os.environ["MASTER_PORT"]
if "RANK" in os.environ: del os.environ["RANK"]
if "WORLD_SIZE" in os.environ: del os.environ["WORLD_SIZE"]
if "LOCAL_RANK" in os.environ: del os.environ["LOCAL_RANK"]

# Setup CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Check bitsandbytes
try:
    import bitsandbytes
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes"])
    import bitsandbytes

if version.parse(bitsandbytes.__version__) < version.parse("0.39.0"):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "bitsandbytes"])
    print("Please restart the runtime after upgrading bitsandbytes.")
    sys.exit(0)

# Install packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate>=0.23.0"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "trl>=0.7.4"])

# Performance settings
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer

print("Setting up QLoRA with model parallel...")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

model_name = "unsloth/meta-Llama-3.1-8B-Instruct-bnb-4bit"
max_seq_length = 1024

# Configure BnB quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load model with device_map to distribute across GPUs
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # This is what provides the multi-GPU support
    torch_dtype=torch.float16,
    offload_folder="offload",  # Enable CPU offloading
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"

model.config.use_cache = False

# Apply LoRA
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

# Set trainable parameters - only LoRA adapters
with torch.no_grad():
    for name, param in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# Load dataset
url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
dataset = load_dataset("json", data_files={"train": url}, split="train[:5%]")

# Training arguments WITHOUT FSDP as it's not compatible with single-process notebook
training_args = TrainingArguments(
    output_dir="./llama-3.1-8b-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=1,
    max_steps=15,
    logging_steps=1,
    save_steps=15,
    learning_rate=2e-5,
    fp16=True,
    optim="adamw_torch",
    report_to="none",
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Print configuration
print(f"Starting training with QLoRA and model parallelism")
print(f"Model: {model_name}")
print(f"Device map: {model.hf_device_map}")
print(f"Dataset size: {len(dataset)}")
print(f"Batch size per device: {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")

# Start training
trainer.train()

# Save model
model.save_pretrained("./llama-3.1-8b-lora")
print("Training complete. Model saved to ./llama-3.1-8b-lora")
