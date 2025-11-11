# Evaluation Summary

## Consumer Contracts QA (LegalBench)
- Base Gemma-3-4B-IT accuracy: **70.7%** (280/396, 95% CI 0.662–0.752)
- LoRA adapter accuracy: **74.0%** (293/396, 95% CI 0.697–0.783)
- Delta: **+3.3 points**, McNemar χ² = 2.087 (p ≈ 0.1486)
- LoRA produced better answers on 41 prompts and regressed on 28, indicating a net improvement with room for further tuning.

## Next Steps
- Evaluate additional LegalBench shards (`contract_nli_*`, `privacy_policy_qa`, `successor_liability`).
- Run MMLU Professional Law using `eval_gemma3_mmlu.py` to monitor retention of general legal knowledge.
- Incorporate structured logging of evaluation runs (JSON/CSV) for automated reporting.
