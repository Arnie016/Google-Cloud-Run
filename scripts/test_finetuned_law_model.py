#!/usr/bin/env python3
"""
Load and test your fine-tuned law model
Fixed version - handles bitsandbytes error
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import glob
import os

print("=" * 60)
print("Loading Fine-Tuned Law Model")
print("=" * 60)

# Find latest checkpoint
checkpoint_dir = "/qwen2.5-32b-law-finetuned"
checkpoints = sorted(glob.glob(f"{checkpoint_dir}/checkpoint-*"))
if checkpoints:
    latest_checkpoint = checkpoints[-1]
    print(f"\n✅ Found checkpoint: {latest_checkpoint}")
else:
    print(f"\n❌ No checkpoints found in {checkpoint_dir}")
    exit(1)

# Load base model (FIXED - use dtype instead of torch_dtype)
print("\n1. Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct",
    trust_remote_code=True
)

# Suppress bitsandbytes warning
import warnings
warnings.filterwarnings('ignore', category=RuntimeError, message='.*bitsandbytes.*')

try:
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-32B-Instruct",
        dtype=torch.bfloat16,  # Use dtype instead of torch_dtype
        device_map="auto",
        trust_remote_code=True
    )
    print("✅ Base model loaded")
except Exception as e:
    print(f"Error loading base model: {e}")
    exit(1)

# Load fine-tuned adapter
print(f"\n2. Loading fine-tuned adapter from {latest_checkpoint}...")
try:
    model = PeftModel.from_pretrained(base_model, latest_checkpoint)
    print("✅ Fine-tuned adapter loaded")
except Exception as e:
    print(f"Error loading adapter: {e}")
    exit(1)

# Test the model
print("\n3. Testing fine-tuned model...")
print("-" * 60)

test_prompts = [
    "What is contract law?",
    "Explain the difference between civil and criminal law.",
    "What are the key elements of a valid contract?",
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n📝 Test {i}: {prompt}")
    print("-" * 40)
    
    # Format prompt
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant response
    if "<|im_start|>assistant\n" in response:
        response = response.split("<|im_start|>assistant\n")[-1]
    
    print(f"🤖 Response:\n{response[:300]}...")

print("\n" + "=" * 60)
print("✅ Model testing complete!")
print("=" * 60)
print(f"\nModel location: {latest_checkpoint}")
print("You can now use this model for legal tasks!")

