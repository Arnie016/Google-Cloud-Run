#!/usr/bin/env python3
"""
Access and use your fine-tuned Qwen 2.5 32B law model
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from pathlib import Path

print("=" * 60)
print("ACCESSING YOUR FINE-TUNED MODEL")
print("=" * 60)

# Check available checkpoints
checkpoint_dir = "/qwen2.5-32b-law-finetuned"
print(f"\n1. Checking checkpoints in: {checkpoint_dir}")

if os.path.exists(checkpoint_dir):
    checkpoints = sorted(Path(checkpoint_dir).glob("checkpoint-*"))
    print(f"✅ Found {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints[-5:]:  # Show last 5
        print(f"   - {ckpt.name}")
    
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        print(f"\n📌 Latest checkpoint: {latest_checkpoint.name}")
    else:
        print("⚠️  No checkpoint directories found")
        latest_checkpoint = None
else:
    print(f"❌ Directory not found: {checkpoint_dir}")
    latest_checkpoint = None

# Load fine-tuned model
print("\n2. Loading fine-tuned model...")

base_model_name = "Qwen/Qwen2.5-32B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

# Load base model
print(f"   Loading base model: {base_model_name}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter
if latest_checkpoint:
    adapter_path = latest_checkpoint
    print(f"   Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("✅ Fine-tuned model loaded!")
else:
    print("⚠️  No checkpoint found, using base model only")
    model = base_model

# Enable inference mode
model.eval()

# Test the model
print("\n3. Testing fine-tuned model...")
print("-" * 60)

test_prompts = [
    "What is contract law?",
    "Explain the difference between civil and criminal law.",
    "What are the key elements of a valid contract?",
]

for prompt in test_prompts:
    print(f"\n📝 Prompt: {prompt}")
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
    
    print(f"🤖 Response: {response[:200]}...")

print("\n" + "=" * 60)
print("✅ MODEL ACCESS COMPLETE!")
print("=" * 60)

print("\n📖 How to use:")
print("   1. Load base model + adapter (as shown above)")
print("   2. Use PeftModel.from_pretrained() to load adapter")
print("   3. Generate responses like normal")
print("\n💡 Tip: Use latest checkpoint for best results")

