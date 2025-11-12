# Evaluation Summary

## Consumer Contracts QA (LegalBench)
- Base Gemma-3-4B-IT accuracy: **70.7%** (280/396, 95% CI 0.662–0.752)
- LoRA adapter accuracy: **74.5%** (295/396, 95% CI 0.702–0.788)
- Delta: **+3.8 points**, McNemar χ² = 2.925 (p ≈ 0.0872)
- LoRA improved 41 prompts and regressed on 26.

## Contract NLI – Confidentiality of Agreement
- Base accuracy: **68.3%** (56/82, 95% CI 0.582–0.784)
- LoRA accuracy: **62.2%** (51/82, 95% CI 0.517–0.727)
- Delta: **−6.1 points**, McNemar χ² = 1.778 (p ≈ 0.1824)
- Regression driven by 7 cases where base was correct and LoRA was not.

## Privacy Policy QA
- Base accuracy: **62.0%** (6776/10923, 95% CI 0.611–0.629)
- LoRA accuracy: **53.4%** (5837/10923, 95% CI 0.525–0.544)
- Delta: **−8.6 points**, McNemar χ² = 126.143 (p ≈ 0.0000)
- LoRA underperformed on 3957 prompts despite 3018 improvements, signalling the need for targeted retraining.

## Successor Liability
- Base accuracy: **21.3%** (10/47, 95% CI 0.096–0.330)
- LoRA accuracy: **21.3%** (10/47, 95% CI 0.096–0.330)
- Delta: **0.0 points**, McNemar χ² = 0.000 (p = 1.0000)
- Models tied; no disagreements across the test questions.

## Next Steps
- Investigate regression on privacy-policy and contract-NLI shards (data filtering, targeted SFT, or GRPO on high-impact prompts).
- Run MMLU Professional Law via `eval_gemma3_mmlu.py` to baseline broader retention.
- Export metrics to a structured CSV/JSON for dashboards in the final submission.
