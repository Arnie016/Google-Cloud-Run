# Demo Plan

1. **Problem & Solution (30s)**
   - Legal teams need faster answers on consumer contract obligations.
   - We fine-tuned Gemma-3-4B-IT with LoRA on curated LegalBench shards.

2. **Architecture Walkthrough (30s)**
   - Slide or diagram: Client → Cloud Run (L4 GPU) → FastAPI → Gemma + LoRA → Response.
   - Highlight logging/monitoring and GCS/Artifact Registry for model assets.

3. **Live Endpoint Call (45s)**
   - Run:
     ```bash
     curl -s -X POST "https://legal-gemma-817958879077.europe-west4.run.app/v1/legal/answer" \
       -H "Content-Type: application/json" \
       -H "X-API-Key: change-me" \
       -d '{"query":"Summarize promissory estoppel doctrine in contract law.","max_new_tokens":256,"temperature":0.2}'
     ```
   - Narrate the structured output (IRAC-like bulleting, legal terminology).

4. **Evaluation Results (45s)**
   - Show snippet from `reports/legalbench_consumer_contracts_qa_compare.log`.
   - Mention +3.3 point lift on LegalBench consumer contracts QA.
   - Optional screenshot of benchmark table.

5. **Call to Action & Links (30s)**
   - Point to GitHub repo + README instructions.
   - Share endpoint link + API key instructions.
   - Mention next steps (additional tasks, MMLU, compare_base fixes).
