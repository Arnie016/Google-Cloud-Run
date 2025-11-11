#!/usr/bin/env python3
"""
Scientific Legal Reasoning Evaluation: Base vs Fine-Tuned Model
Based on evaluation design with 0-5 point scoring per dimension
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import glob
import os
import sys
import json
import re
from typing import Dict, List, Tuple
from statistics import mean, stdev
import pandas as pd

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

# =======================
# 10 Test Questions (from evaluation design)
# =======================
TEST_QUESTIONS = [
    {
        "domain": "Contract Law",
        "question": "When does an offer become irrevocable under UCC § 2-205, and how does this differ from common-law options?",
        "key_terms": ["UCC", "2-205", "irrevocable", "firm offer", "common-law", "options", "consideration"],
    },
    {
        "domain": "Constitutional Law",
        "question": "Explain the strict scrutiny test and give two cases where it was applied.",
        "key_terms": ["strict scrutiny", "compelling interest", "narrowly tailored", "cases", "constitutional"],
    },
    {
        "domain": "Criminal Law",
        "question": "Analyze whether an accomplice who withdraws before the crime occurs is still criminally liable.",
        "key_terms": ["accomplice", "withdrawal", "criminal liability", "actus reus", "mens rea", "renunciation"],
    },
    {
        "domain": "Tort Law",
        "question": "Compare negligence per se and res ipsa loquitur; when does each apply?",
        "key_terms": ["negligence per se", "res ipsa loquitur", "statute", "violation", "inference", "applicable"],
    },
    {
        "domain": "Property Law",
        "question": "Discuss the rule against perpetuities and its modern reforms.",
        "key_terms": ["rule against perpetuities", "perpetuities", "reforms", "wait-and-see", "uniform acts"],
    },
    {
        "domain": "Corporate Law",
        "question": "When can the corporate veil be pierced? Give examples.",
        "key_terms": ["corporate veil", "pierce", "alter ego", "undercapitalization", "fraud", "examples"],
    },
    {
        "domain": "Intellectual Property",
        "question": "Differentiate between copyright and trademark fair-use doctrines.",
        "key_terms": ["copyright", "trademark", "fair use", "fair use doctrine", "four factors", "Lanham Act"],
    },
    {
        "domain": "International Law",
        "question": "Explain state immunity under the FSIA and its exceptions.",
        "key_terms": ["FSIA", "Foreign Sovereign Immunities Act", "state immunity", "exceptions", "commercial activity"],
    },
    {
        "domain": "Evidence Law",
        "question": "Outline the hearsay rule and three major exceptions.",
        "key_terms": ["hearsay", "exceptions", "excited utterance", "business records", "dying declaration", "FRE"],
    },
    {
        "domain": "Legal Ethics",
        "question": "Under the ABA Model Rules, when must a lawyer withdraw from representation?",
        "key_terms": ["ABA Model Rules", "withdraw", "representation", "mandatory", "Rule 1.16", "conflict"],
    },
]

# =======================
# Scoring Criteria (0-5 points each)
# =======================
CRITERIA = ["Knowledge Depth", "Accuracy", "Reasoning", "Completeness"]

def extract_response(text: str) -> str:
    """Extract the actual response from model output"""
    if "<|im_start|>assistant\n" in text:
        return text.split("<|im_start|>assistant\n")[-1]
    return text

def score_knowledge_depth(response: str, key_terms: List[str], question: str) -> float:
    """
    Score Knowledge Depth (0-5):
    - Precise legal terminology
    - Citations of doctrines/statutes
    - Domain-specific knowledge
    """
    response_lower = response.lower()
    question_lower = question.lower()
    
    # Count key terms mentioned
    terms_found = sum(1 for term in key_terms if term.lower() in response_lower)
    term_coverage = min(terms_found / len(key_terms) * 5, 5.0) if key_terms else 0
    
    # Check for citations (statutes, cases, rules)
    citations = len(re.findall(r'(§|Section|UCC|FRE|Rule \d+\.\d+|Model Rule|Case:|v\.|[\d]+ U\.S\.)', response))
    citation_score = min(citations * 0.5, 2.0)
    
    # Check for legal doctrines/principles
    doctrine_words = ["doctrine", "principle", "standard", "test", "rule", "exception", "exception"]
    doctrine_count = sum(1 for word in doctrine_words if word in response_lower)
    doctrine_score = min(doctrine_count * 0.3, 1.5)
    
    # Check depth (length and detail)
    word_count = len(response.split())
    depth_bonus = min(word_count / 100, 1.0)  # Max 1 point for sufficient detail
    
    score = min(term_coverage * 0.4 + citation_score * 0.3 + doctrine_score * 0.2 + depth_bonus * 0.1, 5.0)
    return round(score, 2)

def score_accuracy(response: str, domain: str) -> float:
    """
    Score Accuracy (0-5):
    - Correct legal principles
    - Accurate case references
    - No incorrect statements
    """
    response_lower = response.lower()
    
    # Check for common legal errors (penalize)
    errors = 0
    
    # Domain-specific accuracy checks
    if "contract" in domain.lower():
        if "firm offer" in response_lower and "ucc" not in response_lower:
            errors += 0.5
        if "consideration" in response_lower and "irrevocable" in response_lower and "ucc" not in response_lower:
            errors += 0.3
    
    if "constitutional" in domain.lower():
        if "strict scrutiny" in response_lower and "compelling" not in response_lower:
            errors += 0.5
    
    if "criminal" in domain.lower():
        if "accomplice" in response_lower and "withdrawal" in response_lower:
            # Should mention renunciation/withdrawal rules
            if "renunciation" not in response_lower and "abandonment" not in response_lower:
                errors += 0.3
    
    # Check for definitive legal language (reward)
    definitive_words = ["must", "required", "shall", "cannot", "prohibited", "mandatory"]
    definitive_count = sum(1 for word in definitive_words if word in response_lower)
    definitive_score = min(definitive_count * 0.2, 1.5)
    
    # Check for vague/uncertain language (penalize)
    vague_words = ["maybe", "possibly", "uncertain", "unclear", "perhaps", "might be"]
    vague_count = sum(1 for word in vague_words if word in response_lower)
    vague_penalty = min(vague_count * 0.3, 1.5)
    
    # Base accuracy score
    base_score = 3.0
    score = base_score + definitive_score - errors - vague_penalty
    score = max(0, min(5.0, score))
    return round(score, 2)

def score_reasoning(response: str, question: str) -> float:
    """
    Score Reasoning (0-5):
    - Logical connections
    - Cause-effect relationships
    - Step-by-step analysis
    """
    response_lower = response.lower()
    question_lower = question.lower()
    
    # Check for comparisons (if question asks for comparison)
    has_comparison = False
    if "compare" in question_lower or "differ" in question_lower or "differentiate" in question_lower:
        comparison_words = ["difference", "similar", "unlike", "versus", "while", "whereas", "on the other hand"]
        has_comparison = any(word in response_lower for word in comparison_words)
    
    comparison_score = 1.5 if has_comparison else 0
    
    # Check for cause-effect reasoning
    cause_effect_words = ["because", "therefore", "thus", "consequently", "as a result", "due to", "leads to"]
    cause_effect_count = sum(1 for word in cause_effect_words if word in response_lower)
    cause_effect_score = min(cause_effect_count * 0.4, 1.5)
    
    # Check for step-by-step analysis
    step_words = ["first", "second", "third", "step", "stage", "then", "next", "finally"]
    step_count = sum(1 for word in step_words if word in response_lower)
    step_score = min(step_count * 0.3, 1.0)
    
    # Check for conditional reasoning
    conditional_words = ["if", "when", "unless", "provided that", "in case", "assuming"]
    conditional_count = sum(1 for word in conditional_words if word in response_lower)
    conditional_score = min(conditional_count * 0.3, 1.0)
    
    score = comparison_score + cause_effect_score + step_score + conditional_score
    score = min(5.0, score)
    return round(score, 2)

def score_completeness(response: str, question: str) -> float:
    """
    Score Completeness (0-5):
    - Addresses all parts of the question
    - Provides examples when requested
    - Covers all aspects
    """
    response_lower = response.lower()
    question_lower = question.lower()
    
    # Extract question requirements
    requirements = []
    
    if "when" in question_lower:
        requirements.append("when")
    if "how" in question_lower:
        requirements.append("how")
    if "compare" in question_lower or "differ" in question_lower:
        requirements.append("comparison")
    if "explain" in question_lower:
        requirements.append("explanation")
    if "give" in question_lower or "examples" in question_lower or "cases" in question_lower:
        requirements.append("examples")
    if "analyze" in question_lower:
        requirements.append("analysis")
    if "outline" in question_lower:
        requirements.append("outline")
    if "discuss" in question_lower:
        requirements.append("discussion")
    
    # Check if requirements are addressed
    addressed = 0
    for req in requirements:
        if req == "when":
            if "when" in response_lower or any(word in response_lower for word in ["occurs", "applies", "happens"]):
                addressed += 1
        elif req == "how":
            if "how" in response_lower or any(word in response_lower for word in ["by", "through", "via"]):
                addressed += 1
        elif req == "comparison":
            comparison_words = ["difference", "similar", "unlike", "versus", "while", "whereas"]
            if any(word in response_lower for word in comparison_words):
                addressed += 1
        elif req == "examples":
            example_words = ["example", "instance", "case", "such as", "for example"]
            if any(word in response_lower for word in example_words):
                addressed += 1
        elif req in ["explanation", "analysis", "discussion", "outline"]:
            # These are general - check if response is substantial
            if len(response.split()) > 50:
                addressed += 1
    
    # Check for multiple parts (questions often have multiple parts)
    question_parts = len(re.split(r'\?|and|,', question))
    response_sentences = len(re.split(r'[.!?]+', response))
    
    # Base score
    if len(requirements) > 0:
        completeness_score = (addressed / len(requirements)) * 3.0
    else:
        completeness_score = 2.0
    
    # Bonus for addressing multiple parts
    if response_sentences >= question_parts:
        completeness_score += 1.0
    
    # Bonus for examples/cases when requested
    if "examples" in requirements or "cases" in requirements:
        if any(word in response_lower for word in ["example", "instance", "case", "such as"]):
            completeness_score += 1.0
    
    score = min(5.0, completeness_score)
    return round(score, 2)

def score_response(response: str, question_data: Dict) -> Dict[str, float]:
    """Score response on all criteria"""
    scores = {
        "Knowledge Depth": score_knowledge_depth(response, question_data["key_terms"], question_data["question"]),
        "Accuracy": score_accuracy(response, question_data["domain"]),
        "Reasoning": score_reasoning(response, question_data["question"]),
        "Completeness": score_completeness(response, question_data["question"]),
    }
    scores["Total"] = sum(scores.values())
    return scores

def generate_response(model, tokenizer, prompt: str, device, max_tokens: int = 512) -> str:
    """Generate response from model"""
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,  # Lower temperature for more focused responses
            do_sample=True,
            top_p=0.9,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_response(response)

# =======================
# Main Evaluation
# =======================
print("=" * 80)
print("SCIENTIFIC LEGAL REASONING EVALUATION")
print("Base Model vs Fine-Tuned Model")
print("=" * 80)
print("\nEvaluation Design:")
print("  - 10 diverse legal questions")
print("  - 4 scoring criteria (0-5 points each): Knowledge Depth, Accuracy, Reasoning, Completeness")
print("  - Total: 20 points per question")
print("  - Statistical comparison with significance testing")
print("=" * 80)

# Load models
print("\n[1/3] Loading models...")
with SuppressBitsandbytes():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", trust_remote_code=True)

print("  Loading base model...")
with SuppressBitsandbytes():
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-32B-Instruct",
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
base_model.eval()

print("  Loading fine-tuned model...")
checkpoints = sorted(glob.glob("/qwen2.5-32b-law-finetuned/checkpoint-*"))
if not checkpoints:
    print("ERROR: No fine-tuned checkpoints found!")
    sys.exit(1)
latest_checkpoint = checkpoints[-1]
print(f"  Using checkpoint: {latest_checkpoint}")
fine_tuned_model = PeftModel.from_pretrained(base_model, latest_checkpoint)
fine_tuned_model.eval()

# Run evaluations
print("\n[2/3] Running evaluations on 10 questions...")
results = []

for i, question_data in enumerate(TEST_QUESTIONS, 1):
    print(f"\n  Question {i}/10: {question_data['domain']}")
    print(f"  Q: {question_data['question'][:70]}...")
    
    # Generate responses
    base_response = generate_response(base_model, tokenizer, question_data['question'], base_model.device)
    ft_response = generate_response(fine_tuned_model, tokenizer, question_data['question'], fine_tuned_model.device)
    
    # Score responses
    base_scores = score_response(base_response, question_data)
    ft_scores = score_response(ft_response, question_data)
    
    # Determine winner
    winner = "Fine-Tuned" if ft_scores["Total"] > base_scores["Total"] else "Base" if base_scores["Total"] > ft_scores["Total"] else "Tie"
    
    results.append({
        "Domain": question_data['domain'],
        "Question": question_data['question'],
        "Base_Knowledge_Depth": base_scores["Knowledge Depth"],
        "Base_Accuracy": base_scores["Accuracy"],
        "Base_Reasoning": base_scores["Reasoning"],
        "Base_Completeness": base_scores["Completeness"],
        "Base_Total": base_scores["Total"],
        "FineTuned_Knowledge_Depth": ft_scores["Knowledge Depth"],
        "FineTuned_Accuracy": ft_scores["Accuracy"],
        "FineTuned_Reasoning": ft_scores["Reasoning"],
        "FineTuned_Completeness": ft_scores["Completeness"],
        "FineTuned_Total": ft_scores["Total"],
        "Difference": ft_scores["Total"] - base_scores["Total"],
        "Winner": winner,
        "Base_Response": base_response[:500],  # Store truncated for review
        "FineTuned_Response": ft_response[:500],
    })
    
    print(f"    Base: {base_scores['Total']:.1f}/20 | Fine-Tuned: {ft_scores['Total']:.1f}/20 | "
          f"Δ: {ft_scores['Total'] - base_scores['Total']:+.1f} | Winner: {winner}")

# Create DataFrame
df = pd.DataFrame(results)

# Save to CSV
output_file = "legal_eval_results.csv"
df.to_csv(output_file, index=False)
print(f"\n  Results saved to: {output_file}")

# Calculate statistics
print("\n[3/3] Generating statistical analysis...")

base_totals = df["Base_Total"].tolist()
ft_totals = df["FineTuned_Total"].tolist()
differences = df["Difference"].tolist()

summary = {
    "Base_Avg": mean(base_totals),
    "FineTuned_Avg": mean(ft_totals),
    "Base_Std": stdev(base_totals) if len(base_totals) > 1 else 0,
    "FineTuned_Std": stdev(ft_totals) if len(ft_totals) > 1 else 0,
    "Avg_Difference": mean(differences),
    "Std_Difference": stdev(differences) if len(differences) > 1 else 0,
    "FineTuned_Wins": (df["Winner"] == "Fine-Tuned").sum(),
    "Base_Wins": (df["Winner"] == "Base").sum(),
    "Ties": (df["Winner"] == "Tie").sum(),
}

# Statistical significance (paired t-test approximation)
# If average difference > 2 points and consistent, consider significant
mean_diff = summary["Avg_Difference"]
std_diff = summary["Std_Difference"]
t_statistic = mean_diff / (std_diff / (len(differences) ** 0.5)) if std_diff > 0 else 0
significant = abs(t_statistic) > 2.0 and abs(mean_diff) > 2.0

# Print summary
print("\n" + "=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)
print(f"\n📊 OVERALL PERFORMANCE:")
print(f"   Base Model Average:      {summary['Base_Avg']:.2f} / 20.0 (±{summary['Base_Std']:.2f})")
print(f"   Fine-Tuned Average:      {summary['FineTuned_Avg']:.2f} / 20.0 (±{summary['FineTuned_Std']:.2f})")
print(f"   Average Difference:      {summary['Avg_Difference']:+.2f} (±{summary['Std_Difference']:.2f})")

print(f"\n🏆 WINS:")
print(f"   Fine-Tuned Wins:         {summary['FineTuned_Wins']}/10")
print(f"   Base Model Wins:         {summary['Base_Wins']}/10")
print(f"   Ties:                    {summary['Ties']}/10")

print(f"\n📈 BY CRITERIA:")
for criterion in CRITERIA:
    base_col = f"Base_{criterion.replace(' ', '_')}"
    ft_col = f"FineTuned_{criterion.replace(' ', '_')}"
    base_avg = df[base_col].mean()
    ft_avg = df[ft_col].mean()
    diff = ft_avg - base_avg
    print(f"   {criterion:20s}: Base={base_avg:.2f}, FT={ft_avg:.2f}, Δ={diff:+.2f}")

print(f"\n📚 BY DOMAIN:")
for domain in df["Domain"].unique():
    domain_df = df[df["Domain"] == domain]
    base_avg = domain_df["Base_Total"].mean()
    ft_avg = domain_df["FineTuned_Total"].mean()
    diff = ft_avg - base_avg
    print(f"   {domain:25s}: Base={base_avg:.2f}, FT={ft_avg:.2f}, Δ={diff:+.2f}")

print(f"\n📊 STATISTICAL SIGNIFICANCE:")
print(f"   t-statistic:              {t_statistic:.2f}")
print(f"   Mean difference:          {mean_diff:.2f} points")
print(f"   Significant (|t|>2 & |Δ|>2): {'✅ YES' if significant else '❌ NO'}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
if significant and mean_diff > 0:
    print("✅ FINE-TUNING SIGNIFICANTLY IMPROVED LEGAL REASONING")
    print(f"   Average improvement: {mean_diff:.2f} points ({mean_diff/20*100:.1f}%)")
    print(f"   Consistent improvement: {summary['FineTuned_Wins']}/10 questions")
elif significant and mean_diff < 0:
    print("⚠️  BASE MODEL PERFORMED SIGNIFICANTLY BETTER")
    print(f"   Average difference: {mean_diff:.2f} points")
elif abs(mean_diff) < 2.0:
    print("≈ NO SIGNIFICANT DIFFERENCE")
    print("   Fine-tuning may have changed style but not substantively improved reasoning")
    print(f"   Average difference: {mean_diff:.2f} points (less than 2-point threshold)")
else:
    print(f"⚠️  INCONSISTENT RESULTS")
    print(f"   Average difference: {mean_diff:.2f} points")
    print(f"   Variability: {summary['Std_Difference']:.2f}")
print("=" * 80)

# Print detailed results
print("\n" + "=" * 80)
print("DETAILED RESULTS (Top 3 Questions)")
print("=" * 80)
top_3 = df.nlargest(3, "Difference")[["Domain", "Question", "Base_Total", "FineTuned_Total", "Difference"]]
for idx, row in top_3.iterrows():
    print(f"\n{row['Domain']}:")
    print(f"  Q: {row['Question'][:60]}...")
    print(f"  Base: {row['Base_Total']:.1f} | FT: {row['FineTuned_Total']:.1f} | Δ: {row['Difference']:+.1f}")

print("\n" + "=" * 80)
print("Evaluation complete! Check 'legal_eval_results.csv' for full results.")
print("=" * 80)

