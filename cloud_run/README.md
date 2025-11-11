# GemmaLaw Cloud Run Deployment

This directory packages the Gemma-3 4B LoRA adapter and exposes a FastAPI inference server designed for Google Cloud Run GPU (NVIDIA L4).

## Layout

- `Dockerfile` – GPU-enabled image that installs PyTorch + Transformers, unpacks the LoRA adapter, and runs Uvicorn
- `app/main.py` – FastAPI service providing `GET /healthz` and `POST /v1/legal/answer` with optional base vs. LoRA comparison
- `requirements.txt` – Python dependencies (FastAPI, Transformers, PEFT, Accelerate)
- `deploy.sh` – Helper script to build, push, and deploy to Cloud Run with GPU accelerators
- `models/gemma3-legal-lora-original.tar.gz` – LoRA adapter tarball (unpacked inside the image)

## Prerequisites

1. **Google Cloud CLI** (`gcloud`) authenticated and pointed at your hackathon project.
2. **Artifact Registry** and **Cloud Run** APIs enabled:

   ```bash
   gcloud services enable artifactregistry.googleapis.com run.googleapis.com cloudbuild.googleapis.com
   ```

3. **Hugging Face token** (`HF_TOKEN`) with access to `google/gemma-3-4b-it` (export before running `deploy.sh`).
4. **LoRA tarball** at `models/gemma3-legal-lora-original.tar.gz` (already provided in this repo).

## Build & Deploy

```bash
cd cloud_run
export HF_TOKEN="hf_..."    # required for gated Gemma downloads
export API_KEY="hackathon-secret"
./deploy.sh YOUR_PROJECT_ID legal-gemma europe-west4
```

The script will:

1. Create a regional Artifact Registry named `cloud-run-hackathon` (if missing)
2. Submit the build to Cloud Build
3. Deploy the image to Cloud Run with an NVIDIA L4, 4 vCPUs, 16 GiB RAM, and API key protection

Output shows the deployed URL. Test it with:

```bash
SERVICE_URL=$(gcloud run services describe legal-gemma \
  --region europe-west4 \
  --format='value(status.url)')

curl -s -X POST "$SERVICE_URL/v1/legal/answer" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
        "query": "Summarize promissory estoppel doctrine in contract law.",
        "max_new_tokens": 256,
        "temperature": 0.2,
        "compare_base": true
      }' | jq
```

## Environment Variables

| Variable        | Default                     | Description |
|-----------------|-----------------------------|-------------|
| `API_KEY`       | `change-me`                 | API key required via `X-API-Key` header |
| `BASE_MODEL_ID` | `google/gemma-3-4b-it`      | Hugging Face model ID or local path specified via `BASE_MODEL_PATH` |
| `BASE_MODEL_PATH` | _unset_                   | Optional path when the base model is baked into the image |
| `LORA_PATH`     | `/app/models/lora`          | Where the adapter is unpacked inside the container |
| `HF_TOKEN`      | _unset_                     | Hugging Face token passed at runtime; mandatory for gated model downloads |

If you stage the base Gemma weights in GCS or Artifact Registry, set `BASE_MODEL_PATH` to that local directory path and mount or copy it during build.

## Demo Tips

- Use `compare_base: true` to show side-by-side base vs. LoRA responses in your hackathon video.
- Tail logs for structured latency + token counts:

  ```bash
  gcloud logs tail "resource.type=cloud_run_revision AND resource.labels.service_name=legal-gemma"
  ```

- To stream output tokens, wrap the generation call in a server-side generator and return a `StreamingResponse`. The current setup sends the full completion for simplicity.

## Cleanup

```bash
# Delete service
gcloud run services delete legal-gemma --region europe-west4

# Optionally delete Artifact Registry repository
gcloud artifacts repositories delete cloud-run-hackathon --location europe-west4
```

