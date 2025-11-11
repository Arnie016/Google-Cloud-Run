chmod +x scripts/evaluation/compare_base_vs_grpo.py scripts/evaluation/run_comparison.sh
#!/usr/bin/env python3
"""
Compare Base Model vs GRPO Fine-Tuned Model (checkpoint-1000)
Enhanced with pretty output and progress tracking
"""
import os
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["DISABLE_BITSANDBYTES_AUTO_INSTALL"] = "1"

import torch
from unsloth import FastLanguageModel
from peft import PeftModel
import json
from datetime import datetime
import sys
import time

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    GRAY = '\033[90m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")

def print_section(text):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}▶ {text}{Colors.END}")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.GRAY}ℹ️  {text}{Colors.END}")

def progress_bar(current, total, bar_length=40):
    """Create progress bar"""
    percent = current / total
    filled = int(bar_length * percent)
    bar = '█' * filled + '░' * (bar_length - filled)
    return f"[{bar}] {current}/{total} ({percent*100:.1f}%)"

def print_progress(current, total, text=""):
    """Print progress with bar"""
    bar = progress_bar(current, total)
    print(f"\r{Colors.GRAY}{bar} {text}{Colors.END}", end='', flush=True)
    if current == total:
        print()  # New line when complete

# Legal test questions
TEST_QUESTIONS = [
    "When does an offer become irrevocable under UCC § 2-205, and how does this differ from common-law options?",
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

# Legal reward function (same as training)
def legal_reward_function(response, question):
    import re
    reward = 0.0
    
    # Completeness
    response_len = len(response.strip())
    if response_len > 500:
        reward += 3.0
    elif response_len > 300:
        reward += 2.0
    elif response_len > 100:
        reward += 1.0
    elif response_len < 50:
        reward -= 2.0
    
    # Legal terminology
    legal_terms = [
        'statute', 'case', 'court', 'ruling', 'precedent', 'doctrine',
        'law', 'legal', 'constitutional', 'common law', 'statutory',
        'UCC', 'FSIA', 'ABA', 'Model Rules', 'negligence', 'liability',
        'immunity', 'jurisdiction', 'standing', 'causation', 'damages',
    ]
    found_terms = sum(1 for term in legal_terms if term.lower() in response.lower())
    reward += min(found_terms * 0.3, 4.0)
    
    # Citations
    if re.search(r'§|UCC|FSIA|ABA', response):
        reward += 2.0
    
    # Structure (IRAC)
    if re.search(r'(issue|rule|analysis|conclusion)', response, re.I):
        reward += 3.0
    
    # Analysis depth
    analysis_keywords = ['requires', 'mandates', 'prohibits', 'allows']
    analysis_count = sum(1 for word in analysis_keywords if word.lower() in response.lower())
    reward += min(analysis_count * 0.2, 3.0)
    
    # Question-specific
    question_keywords = re.findall(r'\b\w+\b', question.lower())
    question_keywords = [w for w in question_keywords if len(w) > 4]
    matched_keywords = sum(1 for kw in question_keywords if kw in response.lower())
    if matched_keywords >= 3:
        reward += 2.0
    elif matched_keywords >= 1:
        reward += 1.0
    
    # Penalties
    if response_len < 20:
        reward -= 3.0
    if found_terms == 0:
        reward -= 1.0
    
    reward = max(-5.0, min(20.0, reward))
    return reward

def generate_response(model, tokenizer, prompt):
    """Generate response from model"""
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract assistant response
        if "<|im_start|>assistant\n" in response:
            response = response.split("<|im_start|>assistant\n")[-1].strip()
        
        return response
    except KeyboardInterrupt:
        print_error("Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Generation error: {e}")
        return ""

# ============================================================================
# MAIN COMPARISON
# ============================================================================

print_header("Legal Model Comparison: Base vs GRPO Fine-Tuned")

# Check GPU
print_section("GPU Check")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print_success(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    print_warning("No GPU detected - using CPU (will be slow)")

# Load models
print_section("Loading Models")

print_info("Loading base model (Qwen 2.5 32B)...")
try:
    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-32B-Instruct",
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    print_success("Base model loaded")
except Exception as e:
    print_error(f"Failed to load base model: {e}")
    sys.exit(1)

print_info("Loading GRPO fine-tuned model (checkpoint-1000)...")
checkpoint_path = "/root/scripts/grpo/qwen2.5-32b-law-grpo/checkpoint-1000"

if os.path.exists(checkpoint_path):
    try:
        grpo_model = PeftModel.from_pretrained(base_model, checkpoint_path)
        print_success("GRPO model loaded")
    except Exception as e:
        print_error(f"Failed to load GRPO checkpoint: {e}")
        print_warning("Using base model for GRPO comparison")
        grpo_model = base_model
else:
    print_warning(f"Checkpoint not found at {checkpoint_path}")
    print_warning("Using base model for GRPO comparison")
    grpo_model = base_model

grpo_tokenizer = base_tokenizer

# Run evaluation
print_section("Running Evaluation")
print_info(f"Testing on {len(TEST_QUESTIONS)} legal questions...\n")

results = []
start_time = time.time()

for i, question in enumerate(TEST_QUESTIONS, 1):
    print(f"\n{Colors.BOLD}📝 Question {i}/{len(TEST_QUESTIONS)}{Colors.END}")
    print(f"{Colors.GRAY}   {question[:75]}...{Colors.END}")
    
    # Base model
    print(f"   {Colors.CYAN}[Base]{Colors.END} Generating...", end=' ', flush=True)
    try:
        base_response = generate_response(base_model, base_tokenizer, question)
        base_reward = legal_reward_function(base_response, question)
        print_success(f"Reward: {base_reward:.2f}/20")
    except Exception as e:
        print_error(f"Error: {e}")
        base_response = ""
        base_reward = 0.0
    
    # GRPO model
    print(f"   {Colors.GREEN}[GRPO]{Colors.END} Generating...", end=' ', flush=True)
    try:
        grpo_response = generate_response(grpo_model, grpo_tokenizer, question)
        grpo_reward = legal_reward_function(grpo_response, question)
        print_success(f"Reward: {grpo_reward:.2f}/20")
    except Exception as e:
        print_error(f"Error: {e}")
        grpo_response = ""
        grpo_reward = 0.0
    
    # Compare
    improvement = grpo_reward - base_reward
    winner = "GRPO" if improvement > 0 else "Base" if improvement < 0 else "Tie"
    
    if improvement > 0:
        status = f"{Colors.GREEN}▲ +{improvement:.2f}{Colors.END}"
    elif improvement < 0:
        status = f"{Colors.RED}▼ {improvement:.2f}{Colors.END}"
    else:
        status = f"{Colors.YELLOW}— {improvement:.2f}{Colors.END}"
    
    print(f"   {Colors.BOLD}Result:{Colors.END} {status} | Winner: {winner}")
    
    results.append({
        "question": question,
        "base_reward": base_reward,
        "grpo_reward": grpo_reward,
        "improvement": improvement,
        "base_length": len(base_response),
        "grpo_length": len(grpo_response),
        "winner": winner,
    })
    
    # Progress update
    print_progress(i, len(TEST_QUESTIONS), "Questions completed")

elapsed_time = time.time() - start_time

# Summary Statistics
print_header("Summary Statistics")

base_avg = sum(r["base_reward"] for r in results) / len(results)
grpo_avg = sum(r["grpo_reward"] for r in results) / len(results)
improvement_avg = grpo_avg - base_avg

grpo_wins = sum(1 for r in results if r["winner"] == "GRPO")
base_wins = sum(1 for r in results if r["winner"] == "Base")
ties = sum(1 for r in results if r["winner"] == "Tie")

# Average rewards with visual bar
print(f"\n{Colors.BOLD}📊 Average Rewards:{Colors.END}")
base_bar = int(base_avg / 20 * 30)
grpo_bar = int(grpo_avg / 20 * 30)

print(f"\n   {Colors.CYAN}Base Model:{Colors.END}")
print(f"   {'█' * base_bar}{'░' * (30 - base_bar)} {base_avg:.2f}/20")
print(f"\n   {Colors.GREEN}GRPO Model:{Colors.END}")
print(f"   {'█' * grpo_bar}{'░' * (30 - grpo_bar)} {grpo_avg:.2f}/20")

if improvement_avg > 0:
    print(f"\n   {Colors.BOLD}{Colors.GREEN}Improvement: +{improvement_avg:.2f} points ({improvement_avg/base_avg*100:+.1f}%){Colors.END}")
elif improvement_avg < 0:
    print(f"\n   {Colors.BOLD}{Colors.RED}Change: {improvement_avg:.2f} points ({improvement_avg/base_avg*100:+.1f}%){Colors.END}")
else:
    print(f"\n   {Colors.BOLD}{Colors.YELLOW}No change: {improvement_avg:.2f} points{Colors.END}")

# Wins breakdown
print(f"\n{Colors.BOLD}🏆 Wins Breakdown:{Colors.END}")
print(f"   {Colors.GREEN}GRPO better: {grpo_wins}/{len(results)} ({grpo_wins/len(results)*100:.0f}%){Colors.END}")
print(f"   {Colors.CYAN}Base better:  {base_wins}/{len(results)} ({base_wins/len(results)*100:.0f}%){Colors.END}")
print(f"   {Colors.YELLOW}Ties:         {ties}/{len(results)} ({ties/len(results)*100:.0f}%){Colors.END}")

# Length comparison
print(f"\n{Colors.BOLD}📏 Response Length:{Colors.END}")
base_len_avg = sum(r["base_length"] for r in results) / len(results)
grpo_len_avg = sum(r["grpo_length"] for r in results) / len(results)
print(f"   Base avg: {base_len_avg:.0f} chars")
print(f"   GRPO avg: {grpo_len_avg:.0f} chars")
print(f"   Difference: {grpo_len_avg - base_len_avg:+.0f} chars")

# Performance
print(f"\n{Colors.BOLD}⏱️  Performance:{Colors.END}")
print(f"   Total time: {elapsed_time:.1f}s")
print(f"   Avg per question: {elapsed_time/len(TEST_QUESTIONS):.1f}s")

# Save results
output_file = "/root/scripts/grpo/comparison_results.json"
try:
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "base_avg": base_avg,
            "grpo_avg": grpo_avg,
            "improvement": improvement_avg,
            "grpo_wins": grpo_wins,
            "base_wins": base_wins,
            "ties": ties,
            "elapsed_time": elapsed_time,
            "results": results,
        }, f, indent=2)
    print_success(f"Results saved to: {output_file}")
except Exception as e:
    print_error(f"Failed to save results: {e}")

# Final verdict
print_header("Final Verdict")

if improvement_avg > 1.0:
    print(f"\n{Colors.BOLD}{Colors.GREEN}🎉 EXCELLENT! GRPO training improved the model by {improvement_avg:.2f} points!{Colors.END}")
elif improvement_avg > 0.5:
    print(f"\n{Colors.BOLD}{Colors.GREEN}✅ GOOD! GRPO training improved the model by {improvement_avg:.2f} points.{Colors.END}")
elif improvement_avg > 0:
    print(f"\n{Colors.BOLD}{Colors.YELLOW}⚠️  MINIMAL improvement: {improvement_avg:.2f} points. May need more training.{Colors.END}")
elif improvement_avg == 0:
    print(f"\n{Colors.BOLD}{Colors.YELLOW}⚠️  NO CHANGE. Model performance unchanged.{Colors.END}")
else:
    print(f"\n{Colors.BOLD}{Colors.RED}❌ REGRESSION: Model performance decreased by {abs(improvement_avg):.2f} points.{Colors.END}")

if grpo_wins >= len(results) * 0.7:
    print(f"{Colors.GREEN}   GRPO wins {grpo_wins}/{len(results)} comparisons - strong improvement!{Colors.END}")
elif grpo_wins >= len(results) * 0.5:
    print(f"{Colors.YELLOW}   GRPO wins {grpo_wins}/{len(results)} comparisons - moderate improvement.{Colors.END}")
else:
    print(f"{Colors.RED}   GRPO wins {grpo_wins}/{len(results)} comparisons - needs more work.{Colors.END}")

print(f"\n{Colors.GRAY}{'='*70}{Colors.END}\n")
