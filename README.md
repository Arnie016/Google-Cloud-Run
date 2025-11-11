# Legal Gemma on Google Cloud Run

> Fine-tuned Gemma-3-4B-IT for consumer contract QA and related legal tasks, deployed on GPU-backed Cloud Run.

## Quick Links
- **Try it out:** https://legal-gemma-817958879077.europe-west4.run.app (add header `X-API-Key: change-me`)
- **Evaluation log:** `reports/legalbench_consumer_contracts_qa_compare.log`
- **Deployment script:** `cloud_run/deploy.sh`

## Architecture
- Vast.ai GPU (2× RTX 5090) for LoRA training (bf16, gradient checkpointing)
- Adapter artifacts stored in GCS / Artifact Registry
- Cloud Run (NVIDIA L4, `europe-west4`) hosting FastAPI → Tokenizer → Gemma + LoRA adapter

```
Client → FastAPI (Cloud Run) → Tokenizer → Gemma-3-4B + LoRA → Response
              │
              └─ Cloud Logging / Monitoring
```

## Dataset
| Shard | Source | Purpose |
|-------|--------|---------|
| `consumer_contracts_qa` | LegalBench | QA baseline / evaluation |
| `contract_nli_*` | LegalBench | Clause entailment evaluation |
| `privacy_policy_qa` | LegalBench | Privacy compliance QA |
| `successor_liability` | LegalBench | Successor liability reasoning |

Combined SFT dataset (24,423 examples) prepared offline; see `scripts/data/` for cleaning heuristics. Raw JSONLs are not committed.

## Training
- Script: `scripts/training/finetune_qwen_law_single_gpu.py`
- Base model: `google/gemma-3-4b-it`
- LoRA config: r=32, α=64, dropout 0.05, target modules Q/K/V/O/gate/up/down
- Training recipe: bf16, batch size 1, grad accumulation 32, cosine LR 1e-4
- Logs stored under `/root/logs/...` (summarized in docs)

## Inference
- FastAPI server (`cloud_run/app/main.py`)
- `/v1/legal/answer` POST endpoint with optional `compare_base` flag (beta)

Example request:
```
curl -s -X POST "https://legal-gemma-817958879077.europe-west4.run.app/v1/legal/answer" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" \
  -d '{"query":"Summarize promissory estoppel doctrine in contract law.","max_new_tokens":256,"temperature":0.2}'
```

## Evaluation
| Task | Base Acc | LoRA Acc | Δ | Log |
|------|----------|----------|---|-----|
| consumer_contracts_qa | 0.707 | 0.740 | +0.033 | `reports/legalbench_consumer_contracts_qa_compare.log` |

Additional tasks to be added after running `eval_gemma3_legalbench.py` with the respective `TASK` value.

## Demo Plan
1. Hit the Cloud Run endpoint live (show structured LoRA output)
2. Display a snippet from `reports/legalbench_consumer_contracts_qa_compare.log`
3. Highlight architecture diagram and deployment steps
4. Provide repo + endpoint links

## Local Setup
```
pip install -r cloud_run/requirements.txt
fastapi dev cloud_run/app/main.py
```

## Deployment
```
gcloud auth login
gcloud config set project <project-id>
./cloud_run/deploy.sh <project-id> legal-gemma europe-west4
```

## License
Add an appropriate license if you plan to open source the code.
