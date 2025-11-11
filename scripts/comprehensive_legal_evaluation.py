#!/usr/bin/env python3
"""
Comprehensive Legal Knowledge Evaluation: Base vs Fine-Tuned Model

PROBLEM STATEMENT:
==================
We fine-tuned Qwen 2.5 32B on a law dataset (pile_of_law_subset), but we need to determine:
1. Did fine-tuning actually IMPROVE legal knowledge, or just change formatting?
2. Is the fine-tuned model better at complex legal reasoning?
3. Does it retain general knowledge while gaining legal expertise?
4. Are improvements consistent across different legal domains?

TASK:
=====
Evaluate both models on 10 diverse, complex legal questions covering:
- Contract Law (foundation)
- Constitutional Law (complex principles)
- Criminal Law (specific cases)
- Tort Law (liability)
- Property Law (ownership)
- Corporate Law (business)
- Intellectual Property (specialized)
- International Law (broad scope)
- Evidence Law (procedural)
- Legal Ethics (judgment)

For each question, evaluate:
- Knowledge depth (how much detail)
- Accuracy (correct legal concepts)
- Structure (clear organization)
- Reasoning (ability to apply principles)
- Completeness (covers all aspects)

Then compare:
- Which model performs better overall?
- Are improvements consistent or inconsistent?
- What are the strengths/weaknesses of each?
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import glob
import os
import sys
import re
from typing import Dict, List, Tuple

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
# Test Questions (10 Diverse Legal Topics)
# =======================
TEST_QUESTIONS = [
    {
        "category": "Contract Law",
        "difficulty": "Intermediate",
        "question": "What are the essential elements required for a contract to be legally enforceable, and how does the doctrine of consideration differ from promissory estoppel?",
        "key_concepts": ["offer", "acceptance", "consideration", "promissory estoppel", "legal capacity", "legality"],
    },
    {
        "category": "Constitutional Law",
        "difficulty": "Advanced",
        "question": "Explain the difference between strict scrutiny, intermediate scrutiny, and rational basis review in constitutional law, and provide an example of when each standard applies.",
        "key_concepts": ["strict scrutiny", "intermediate scrutiny", "rational basis", "equal protection", "due process"],
    },
    {
        "category": "Criminal Law",
        "difficulty": "Intermediate",
        "question": "What is the difference between murder and manslaughter? Explain the mens rea requirements for each and how voluntary vs involuntary manslaughter differ.",
        "key_concepts": ["murder", "manslaughter", "mens rea", "malice aforethought", "recklessness"],
    },
    {
        "category": "Tort Law",
        "difficulty": "Intermediate",
        "question": "Explain the elements of negligence and how comparative negligence differs from contributory negligence. When would strict liability apply instead?",
        "key_concepts": ["negligence", "duty", "breach", "causation", "damages", "comparative negligence", "strict liability"],
    },
    {
        "category": "Property Law",
        "difficulty": "Advanced",
        "question": "What is the difference between fee simple absolute and life estate? Explain how easements and covenants differ, and when adverse possession may apply.",
        "key_concepts": ["fee simple", "life estate", "easement", "covenant", "adverse possession", "title"],
    },
    {
        "category": "Corporate Law",
        "difficulty": "Intermediate",
        "question": "What are the fiduciary duties of corporate directors? Explain the business judgment rule and when directors might be held liable for breach of duty.",
        "key_concepts": ["fiduciary duty", "duty of care", "duty of loyalty", "business judgment rule", "corporate governance"],
    },
    {
        "category": "Intellectual Property",
        "difficulty": "Intermediate",
        "question": "What is the difference between copyright, trademark, and patent? Explain what constitutes fair use in copyright law.",
        "key_concepts": ["copyright", "trademark", "patent", "fair use", "intellectual property", "infringement"],
    },
    {
        "category": "International Law",
        "difficulty": "Advanced",
        "question": "Explain the principle of sovereign immunity and how it applies in international commercial disputes. What is the difference between absolute and restrictive immunity?",
        "key_concepts": ["sovereign immunity", "absolute immunity", "restrictive immunity", "international law", "jurisdiction"],
    },
    {
        "category": "Evidence Law",
        "difficulty": "Intermediate",
        "question": "What is the hearsay rule and what are the main exceptions? Explain when expert testimony is admissible and what standards apply.",
        "key_concepts": ["hearsay", "exceptions", "expert testimony", "admissibility", "Federal Rules of Evidence", "Daubert"],
    },
    {
        "category": "Legal Ethics",
        "difficulty": "Advanced",
        "question": "What is attorney-client privilege and when can it be waived? Explain the conflict of interest rules and when a lawyer must withdraw from representation.",
        "key_concepts": ["attorney-client privilege", "conflict of interest", "confidentiality", "withdrawal", "ethics"],
    },
]

# =======================
# Evaluation Functions
# =======================
def extract_response(text: str) -> str:
    """Extract the actual response from model output"""
    if "<|im_start|>assistant\n" in text:
        return text.split("<|im_start|>assistant\n")[-1]
    return text

def evaluate_knowledge_depth(response: str, key_concepts: List[str]) -> Dict[str, float]:
    """Evaluate how well response covers key concepts"""
    response_lower = response.lower()
    
    # Count concept mentions
    concepts_mentioned = sum(1 for concept in key_concepts if concept.lower() in response_lower)
    concept_coverage = concepts_mentioned / len(key_concepts) if key_concepts else 0
    
    # Measure depth (length, detail)
    word_count = len(response.split())
    sentence_count = len(re.split(r'[.!?]+', response))
    
    # Check for structured format (lists, numbering)
    has_structure = bool(re.search(r'\d+[\.\)]|[-•]', response[:500]))
    
    # Check for legal citations or examples
    has_examples = bool(re.search(r'example|for instance|such as|case|court', response_lower[:500]))
    
    return {
        "concept_coverage": concept_coverage,
        "concepts_mentioned": concepts_mentioned,
        "total_concepts": len(key_concepts),
        "word_count": word_count,
        "sentence_count": sentence_count,
        "has_structure": has_structure,
        "has_examples": has_examples,
        "depth_score": (concept_coverage * 0.4 + min(word_count / 200, 1.0) * 0.3 + 
                        min(sentence_count / 10, 1.0) * 0.2 + (has_structure * 0.05 + has_examples * 0.05)),
    }

def evaluate_accuracy(response: str, category: str) -> Dict[str, float]:
    """Evaluate accuracy of legal concepts"""
    response_lower = response.lower()
    
    # Check for common legal errors
    errors = 0
    if "contract" in category.lower():
        if "offer" in response_lower and "acceptance" not in response_lower[:500]:
            errors += 0.5
        if "consideration" not in response_lower and "promissory estoppel" not in response_lower:
            errors += 0.3
    
    # Check for vague language
    vague_words = ["maybe", "possibly", "might", "could be", "perhaps", "uncertain"]
    vague_count = sum(1 for word in vague_words if word in response_lower[:500])
    
    # Check for definitive legal language
    definitive_words = ["must", "required", "essential", "element", "principle", "rule"]
    definitive_count = sum(1 for word in definitive_words if word in response_lower[:500])
    
    accuracy_score = max(0, 1.0 - (errors * 0.2) - (vague_count * 0.05) + (definitive_count * 0.02))
    
    return {
        "errors": errors,
        "vague_language": vague_count,
        "definitive_language": definitive_count,
        "accuracy_score": min(1.0, accuracy_score),
    }

def evaluate_reasoning(response: str) -> Dict[str, float]:
    """Evaluate reasoning and analytical quality"""
    response_lower = response.lower()
    
    # Check for comparison/contrast
    comparison_words = ["difference", "compare", "contrast", "versus", "unlike", "similar"]
    has_comparison = any(word in response_lower for word in comparison_words)
    
    # Check for cause-effect reasoning
    cause_effect_words = ["because", "therefore", "thus", "consequently", "due to", "result"]
    has_cause_effect = any(word in response_lower for word in cause_effect_words)
    
    # Check for conditional reasoning
    conditional_words = ["if", "when", "unless", "provided that", "in case"]
    has_conditional = any(word in response_lower for word in conditional_words)
    
    # Check for examples
    example_words = ["example", "instance", "case", "illustration"]
    has_examples = any(word in response_lower for word in example_words)
    
    reasoning_score = (
        (has_comparison * 0.3 + has_cause_effect * 0.25 + 
         has_conditional * 0.25 + has_examples * 0.2)
    )
    
    return {
        "has_comparison": has_comparison,
        "has_cause_effect": has_cause_effect,
        "has_conditional": has_conditional,
        "has_examples": has_examples,
        "reasoning_score": reasoning_score,
    }

def evaluate_completeness(response: str, question: str) -> Dict[str, float]:
    """Evaluate if response fully addresses the question"""
    question_lower = question.lower()
    response_lower = response.lower()
    
    # Extract question parts
    question_keywords = []
    if "difference" in question_lower:
        question_keywords.append("difference")
    if "explain" in question_lower:
        question_keywords.append("explain")
    if "how" in question_lower:
        question_keywords.append("how")
    if "when" in question_lower:
        question_keywords.append("when")
    if "what" in question_lower:
        question_keywords.append("what")
    
    # Check if response addresses question keywords
    addresses_question = sum(1 for keyword in question_keywords if keyword in response_lower) / max(len(question_keywords), 1)
    
    # Check for multiple parts (questions often have multiple parts)
    question_parts = len(re.split(r'\?|and|,', question))
    response_parts = len(re.split(r'[.!?]', response))
    
    completeness_score = (addresses_question * 0.5 + min(response_parts / question_parts, 1.0) * 0.5)
    
    return {
        "addresses_question": addresses_question,
        "question_parts": question_parts,
        "response_parts": response_parts,
        "completeness_score": completeness_score,
    }

def generate_response(model, tokenizer, prompt: str, device, max_tokens: int = 512) -> str:
    """Generate response from model"""
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_response(response)

# =======================
# Main Evaluation
# =======================
print("=" * 80)
print("COMPREHENSIVE LEGAL KNOWLEDGE EVALUATION")
print("Base Model vs Fine-Tuned Model")
print("=" * 80)
print("\nPROBLEM:")
print("  - Fine-tuned on law dataset, but unclear if knowledge improved")
print("  - Need to test across diverse legal domains")
print("  - Evaluate depth, accuracy, reasoning, completeness")
print("\nTASK:")
print("  - Test 10 diverse legal questions")
print("  - Compare base vs fine-tuned responses")
print("  - Measure multiple quality dimensions")
print("  - Determine if fine-tuning actually improved legal knowledge")
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

for i, test_case in enumerate(TEST_QUESTIONS, 1):
    print(f"\n  Question {i}/10: {test_case['category']} ({test_case['difficulty']})")
    print(f"  Q: {test_case['question'][:80]}...")
    
    # Generate responses
    base_response = generate_response(base_model, tokenizer, test_case['question'], base_model.device)
    ft_response = generate_response(fine_tuned_model, tokenizer, test_case['question'], fine_tuned_model.device)
    
    # Evaluate
    base_knowledge = evaluate_knowledge_depth(base_response, test_case['key_concepts'])
    ft_knowledge = evaluate_knowledge_depth(ft_response, test_case['key_concepts'])
    
    base_accuracy = evaluate_accuracy(base_response, test_case['category'])
    ft_accuracy = evaluate_accuracy(ft_response, test_case['category'])
    
    base_reasoning = evaluate_reasoning(base_response)
    ft_reasoning = evaluate_reasoning(ft_response)
    
    base_completeness = evaluate_completeness(base_response, test_case['question'])
    ft_completeness = evaluate_completeness(ft_response, test_case['question'])
    
    # Overall score
    base_overall = (
        base_knowledge['depth_score'] * 0.3 +
        base_accuracy['accuracy_score'] * 0.3 +
        base_reasoning['reasoning_score'] * 0.2 +
        base_completeness['completeness_score'] * 0.2
    )
    
    ft_overall = (
        ft_knowledge['depth_score'] * 0.3 +
        ft_accuracy['accuracy_score'] * 0.3 +
        ft_reasoning['reasoning_score'] * 0.2 +
        ft_completeness['completeness_score'] * 0.2
    )
    
    results.append({
        "category": test_case['category'],
        "difficulty": test_case['difficulty'],
        "base": {
            "response": base_response,
            "knowledge": base_knowledge,
            "accuracy": base_accuracy,
            "reasoning": base_reasoning,
            "completeness": base_completeness,
            "overall": base_overall,
        },
        "fine_tuned": {
            "response": ft_response,
            "knowledge": ft_knowledge,
            "accuracy": ft_accuracy,
            "reasoning": ft_reasoning,
            "completeness": ft_completeness,
            "overall": ft_overall,
        },
    })
    
    print(f"    Base: {base_overall:.2f} | Fine-tuned: {ft_overall:.2f} | "
          f"{'✅ FT Better' if ft_overall > base_overall else '✅ Base Better' if base_overall > ft_overall else '≈ Tie'}")

# Generate report
print("\n[3/3] Generating comprehensive report...")
print("\n" + "=" * 80)
print("DETAILED RESULTS")
print("=" * 80)

for i, result in enumerate(results, 1):
    print(f"\n{'='*80}")
    print(f"QUESTION {i}: {result['category']} ({result['difficulty']})")
    print(f"{'='*80}")
    print(f"\n📊 OVERALL SCORES:")
    print(f"   Base Model:      {result['base']['overall']:.3f}")
    print(f"   Fine-Tuned:     {result['fine_tuned']['overall']:.3f}")
    print(f"   Difference:     {result['fine_tuned']['overall'] - result['base']['overall']:+.3f}")
    
    print(f"\n📚 KNOWLEDGE DEPTH:")
    print(f"   Base:      Concepts: {result['base']['knowledge']['concepts_mentioned']}/{result['base']['knowledge']['total_concepts']} "
          f"({result['base']['knowledge']['concept_coverage']*100:.1f}%) | "
          f"Words: {result['base']['knowledge']['word_count']} | "
          f"Depth: {result['base']['knowledge']['depth_score']:.3f}")
    print(f"   Fine-Tuned: Concepts: {result['fine_tuned']['knowledge']['concepts_mentioned']}/{result['fine_tuned']['knowledge']['total_concepts']} "
          f"({result['fine_tuned']['knowledge']['concept_coverage']*100:.1f}%) | "
          f"Words: {result['fine_tuned']['knowledge']['word_count']} | "
          f"Depth: {result['fine_tuned']['knowledge']['depth_score']:.3f}")
    
    print(f"\n✅ ACCURACY:")
    print(f"   Base:      Score: {result['base']['accuracy']['accuracy_score']:.3f} | "
          f"Definitive language: {result['base']['accuracy']['definitive_language']}")
    print(f"   Fine-Tuned: Score: {result['fine_tuned']['accuracy']['accuracy_score']:.3f} | "
          f"Definitive language: {result['fine_tuned']['accuracy']['definitive_language']}")
    
    print(f"\n🧠 REASONING:")
    print(f"   Base:      Score: {result['base']['reasoning']['reasoning_score']:.3f} | "
          f"Comparison: {result['base']['reasoning']['has_comparison']} | "
          f"Examples: {result['base']['reasoning']['has_examples']}")
    print(f"   Fine-Tuned: Score: {result['fine_tuned']['reasoning']['reasoning_score']:.3f} | "
          f"Comparison: {result['fine_tuned']['reasoning']['has_comparison']} | "
          f"Examples: {result['fine_tuned']['reasoning']['has_examples']}")
    
    print(f"\n📋 COMPLETENESS:")
    print(f"   Base:      Score: {result['base']['completeness']['completeness_score']:.3f}")
    print(f"   Fine-Tuned: Score: {result['fine_tuned']['completeness']['completeness_score']:.3f}")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

base_scores = [r['base']['overall'] for r in results]
ft_scores = [r['fine_tuned']['overall'] for r in results]
differences = [ft - base for base, ft in zip(base_scores, ft_scores)]

print(f"\n📊 OVERALL PERFORMANCE:")
print(f"   Base Model Average:      {sum(base_scores)/len(base_scores):.3f}")
print(f"   Fine-Tuned Average:      {sum(ft_scores)/len(ft_scores):.3f}")
print(f"   Average Improvement:     {sum(differences)/len(differences):+.3f}")

print(f"\n🏆 WINS:")
base_wins = sum(1 for d in differences if d < 0)
ft_wins = sum(1 for d in differences if d > 0)
ties = sum(1 for d in differences if abs(d) < 0.01)
print(f"   Base Model Wins:        {base_wins}/10")
print(f"   Fine-Tuned Wins:         {ft_wins}/10")
print(f"   Ties:                    {ties}/10")

print(f"\n📈 BY DIFFICULTY:")
difficulties = {}
for r in results:
    diff = r['difficulty']
    if diff not in difficulties:
        difficulties[diff] = {'base': [], 'ft': []}
    difficulties[diff]['base'].append(r['base']['overall'])
    difficulties[diff]['ft'].append(r['fine_tuned']['overall'])

for diff, scores in difficulties.items():
    base_avg = sum(scores['base']) / len(scores['base'])
    ft_avg = sum(scores['ft']) / len(scores['ft'])
    print(f"   {diff}: Base={base_avg:.3f}, FT={ft_avg:.3f}, Δ={ft_avg-base_avg:+.3f}")

print(f"\n📚 BY CATEGORY:")
category_performance = {}
for r in results:
    cat = r['category']
    if cat not in category_performance:
        category_performance[cat] = {'base': [], 'ft': []}
    category_performance[cat]['base'].append(r['base']['overall'])
    category_performance[cat]['ft'].append(r['fine_tuned']['overall'])

for cat, scores in sorted(category_performance.items()):
    base_avg = sum(scores['base']) / len(scores['base'])
    ft_avg = sum(scores['ft']) / len(scores['ft'])
    print(f"   {cat:25s}: Base={base_avg:.3f}, FT={ft_avg:.3f}, Δ={ft_avg-base_avg:+.3f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
if sum(differences) / len(differences) > 0.05:
    print("✅ FINE-TUNING IMPROVED LEGAL KNOWLEDGE")
    print(f"   Average improvement: {sum(differences)/len(differences):.3f}")
elif sum(differences) / len(differences) < -0.05:
    print("⚠️  BASE MODEL PERFORMED BETTER")
    print(f"   Average difference: {sum(differences)/len(differences):.3f}")
else:
    print("≈ SIMILAR PERFORMANCE")
    print("   Fine-tuning may have changed style but not significantly improved knowledge")
print("=" * 80)

