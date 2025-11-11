#!/usr/bin/env python3
"""
Three-Way Comparison: Base vs OLD (500 steps) vs NEW (10k steps)
Run this after NEW training completes
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from statistics import mean, stdev
import sys

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
OLD_MODEL_PATH = "./qwen2.5-32b-law-finetuned-500steps"  # From previous training
NEW_MODEL_PATH = "./qwen2.5-32b-law-finetuned"  # From new 10k-step training

TEST_QUESTIONS = [
    "When does an offer become irrevocable under UCC § 2-205?",
    "Explain the strict scrutiny test and give two cases where it was applied.",
    "Analyze whether an accomplice who withdraws before the crime occurs is still criminally liable.",
    "Compare negligence per se and res ipsa loquitur; when does each apply?",
    "Discuss the rule against perpetuities and its modern reforms.",
    "When can the corporate veil be pierced? Give examples.",
    "Differentiate between copyright and trademark fair-use doctrines.",
    "Explain state immunity under the FSIA and its exceptions.",
    "Outline the hearsay rule and three major exceptions.",
    "Under the ABA Model Rules, when must a lawyer withdraw from representation?",
]

def load_model(model_path, is_peft=False):
    """Load model (base or fine-tuned)"""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    if is_peft:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        return model, tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True
        )
        return model, tokenizer

def generate_response(model, tokenizer, question):
    """Generate response from model"""
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

def simple_score(response, question):
    """Simple scoring (0-20 points)"""
    score = 0.0
    
    # Length check
    if len(response) > 100:
        score += 5
    elif len(response) > 50:
        score += 3
    
    # Legal terms
    legal_terms = ['law', 'legal', 'statute', 'case', 'court', 'judge', 'ruling', 'precedent']
    found_terms = sum(1 for term in legal_terms if term.lower() in response.lower())
    score += min(found_terms * 1.5, 10)
    
    # Structure (has paragraphs)
    if '\n\n' in response or len(response.split('.')) > 3:
        score += 5
    
    return min(score, 20)

print("=" * 80)
print("THREE-WAY MODEL COMPARISON")
print("=" * 80)
print("\nLoading models...")

# Load models
print("1. Loading base model...")
base_model, tokenizer = load_model(BASE_MODEL, is_peft=False)

print("2. Loading OLD fine-tuned model (500 steps)...")
try:
    old_model, _ = load_model(OLD_MODEL_PATH, is_peft=True)
    old_available = True
except Exception as e:
    print(f"   ⚠️  OLD model not found: {e}")
    old_available = False

print("3. Loading NEW fine-tuned model (10k steps)...")
try:
    new_model, _ = load_model(NEW_MODEL_PATH, is_peft=True)
    new_available = True
except Exception as e:
    print(f"   ⚠️  NEW model not found: {e}")
    new_available = False

if not new_available:
    print("\n❌ NEW model not found. Train first!")
    sys.exit(1)

print("\n" + "=" * 80)
print("EVALUATING MODELS")
print("=" * 80)

results = []

for i, question in enumerate(TEST_QUESTIONS, 1):
    print(f"\nQuestion {i}/10: {question[:60]}...")
    
    # Generate responses
    base_response = generate_response(base_model, tokenizer, question)
    base_score = simple_score(base_response, question)
    
    new_response = generate_response(new_model, tokenizer, question)
    new_score = simple_score(new_response, question)
    
    old_score = None
    if old_available:
        old_response = generate_response(old_model, tokenizer, question)
        old_score = simple_score(old_response, question)
    
    results.append({
        "Question": question,
        "Base_Score": base_score,
        "OLD_Score": old_score if old_available else None,
        "NEW_Score": new_score,
        "NEW_vs_Base": new_score - base_score,
        "NEW_vs_OLD": new_score - old_score if old_available else None,
    })
    
    print(f"  Base: {base_score:.1f}/20", end="")
    if old_available:
        print(f" | OLD: {old_score:.1f}/20", end="")
    print(f" | NEW: {new_score:.1f}/20 | Δ: {new_score - base_score:+.1f}")

# Create DataFrame
df = pd.DataFrame(results)

# Save results
output_file = "three_way_comparison.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ Results saved to: {output_file}")

# Statistics
print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS")
print("=" * 80)

base_scores = df["Base_Score"].tolist()
new_scores = df["NEW_Score"].tolist()
differences = df["NEW_vs_Base"].tolist()

print(f"\nBase Model Average:     {mean(base_scores):.2f}/20")
print(f"NEW Model Average:      {mean(new_scores):.2f}/20")
print(f"Average Improvement:    {mean(differences):+.2f} points")

if old_available:
    old_scores = [s for s in df["OLD_Score"].tolist() if s is not None]
    print(f"OLD Model Average:      {mean(old_scores):.2f}/20")
    print(f"NEW vs OLD Improvement: {mean(new_scores) - mean(old_scores):+.2f} points")

# t-statistic
std_diff = stdev(differences) if len(differences) > 1 else 0
mean_diff = mean(differences)
t_statistic = mean_diff / (std_diff / (len(differences) ** 0.5)) if std_diff > 0 else 0
significant = abs(t_statistic) > 2.0 and abs(mean_diff) > 2.0

print(f"\nt-statistic:             {t_statistic:.2f}")
print(f"Significant (|t|>2 & |Δ|>2): {'✅ YES' if significant else '❌ NO'}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
if significant and mean_diff > 0:
    print("✅ NEW MODEL (10k steps) SIGNIFICANTLY BETTER THAN BASE")
    print(f"   Improvement: {mean_diff:.2f} points ({mean_diff/20*100:.1f}%)")
elif mean_diff > 0:
    print("✅ NEW MODEL BETTER, but not statistically significant")
    print(f"   Improvement: {mean_diff:.2f} points")
else:
    print("❌ NEW MODEL NOT BETTER THAN BASE")
    print(f"   Difference: {mean_diff:.2f} points")
print("=" * 80)
















