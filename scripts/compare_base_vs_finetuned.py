#!/usr/bin/env python3
"""
Compare Base Model vs Fine-Tuned Model
See if fine-tuning improved legal knowledge
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import glob
import os
import sys

# Suppress bitsandbytes errors
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

class SuppressBitsandbytes:
    def __enter__(self):
        self.stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    def __exit__(self, *args):
        sys.stderr.close()
        sys.stderr = self.stderr

print("=" * 60)
print("Comparing Base Model vs Fine-Tuned Model")
print("=" * 60)

# Test prompt
prompt = "What is contract law?"
formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

# Load base model
print("\n1. Loading BASE model (no fine-tuning)...")
with SuppressBitsandbytes():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", trust_remote_code=True)

with SuppressBitsandbytes():
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-32B-Instruct",
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

inputs = tokenizer(formatted, return_tensors="pt").to(base_model.device)

with torch.no_grad():
    base_outputs = base_model.generate(**inputs, max_new_tokens=256, temperature=0.7)

base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
if "<|im_start|>assistant\n" in base_response:
    base_response = base_response.split("<|im_start|>assistant\n")[-1]

print("\n" + "=" * 60)
print("BASE MODEL RESPONSE (No Fine-Tuning):")
print("=" * 60)
print(base_response[:500])

# Load fine-tuned model
print("\n\n2. Loading FINE-TUNED model (trained on law dataset)...")
checkpoints = sorted(glob.glob("/qwen2.5-32b-law-finetuned/checkpoint-*"))
latest = checkpoints[-1]
print(f"Using checkpoint: {latest}")

fine_tuned_model = PeftModel.from_pretrained(base_model, latest)

inputs = tokenizer(formatted, return_tensors="pt").to(fine_tuned_model.device)

with torch.no_grad():
    ft_outputs = fine_tuned_model.generate(**inputs, max_new_tokens=256, temperature=0.7)

ft_response = tokenizer.decode(ft_outputs[0], skip_special_tokens=True)
if "<|im_start|>assistant\n" in ft_response:
    ft_response = ft_response.split("<|im_start|>assistant\n")[-1]

print("\n" + "=" * 60)
print("FINE-TUNED MODEL RESPONSE (Trained on Law Dataset):")
print("=" * 60)
print(ft_response[:500])

# Compare
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print("\n📊 Response Length:")
print(f"   Base model: {len(base_response)} chars")
print(f"   Fine-tuned: {len(ft_response)} chars")

print("\n📚 Legal Terms Mentioned:")
legal_terms = ['offer', 'acceptance', 'consideration', 'breach', 'jurisdiction', 'obligation', 'remedy']
base_terms = [term for term in legal_terms if term.lower() in base_response.lower()]
ft_terms = [term for term in legal_terms if term.lower() in ft_response.lower()]
print(f"   Base model: {len(base_terms)} terms - {base_terms}")
print(f"   Fine-tuned: {len(ft_terms)} terms - {ft_terms}")

print("\n🎯 Structured Format:")
base_structured = "1." in base_response or "2." in base_response or "-" in base_response[:100]
ft_structured = "1." in ft_response or "2." in ft_response or "-" in ft_response[:100]
print(f"   Base model: {'✅ Structured' if base_structured else '❌ Not structured'}")
print(f"   Fine-tuned: {'✅ Structured' if ft_structured else '❌ Not structured'}")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
if len(ft_terms) > len(base_terms) or ft_structured:
    print("✅ Fine-tuning improved the response!")
    print("   The model learned from the law dataset.")
else:
    print("⚠️  Similar responses - may need more training")
print("=" * 60)

