#!/usr/bin/env bash
# Build and deploy the GemmaLaw Cloud Run GPU service.
set -euo pipefail

if [[ $# < 2 ]]; then
  echo "Usage: $0 PROJECT_ID SERVICE_NAME [REGION]" >&2
  exit 1
fi

PROJECT_ID="$1"
SERVICE_NAME="$2"
REGION="${3:-europe-west4}"
REPO_NAME="${REPO_NAME:-cloud-run-hackathon}"
API_KEY_ENV="${API_KEY:-change-me}"
BASE_MODEL_ID="${BASE_MODEL_ID:-google/gemma-3-4b-it}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:$(date +%Y%m%d%H%M%S)"

# Ensure Artifact Registry exists
if ! gcloud artifacts repositories describe "${REPO_NAME}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" >/dev/null 2>&1; then
  echo ":: Creating Artifact Registry ${REPO_NAME} in ${REGION}."
  gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=Docker \
    --project "${PROJECT_ID}" \
    --location "${REGION}" \
    --description="Cloud Run GPU images"
fi

# Prepare isolated build context to avoid iCloud timeouts
BUILD_CONTEXT="$(mktemp -d -t gemmalaw-build-XXXXXX)"
cleanup() {
  rm -rf "${BUILD_CONTEXT}"
}
trap cleanup EXIT

mkdir -p "${BUILD_CONTEXT}/cloud_run" "${BUILD_CONTEXT}/models"
rsync -a "${SCRIPT_DIR}/" "${BUILD_CONTEXT}/cloud_run/"
cp "${REPO_ROOT}/models/gemma3-legal-lora-original.tar.gz" "${BUILD_CONTEXT}/models/"

echo ":: Building container ${IMAGE}."
gcloud builds submit "${BUILD_CONTEXT}" \
  --config "${BUILD_CONTEXT}/cloud_run/cloudbuild.yaml" \
  --substitutions=_IMAGE="${IMAGE}"

deploy_cmd=(
  gcloud alpha run deploy "${SERVICE_NAME}" \
    --image="${IMAGE}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --platform=managed \
    --allow-unauthenticated \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --cpu=4 \
    --memory=16Gi \
    --timeout=900 \
    --concurrency=1 \
    --max-instances=1 \
    --set-env-vars="API_KEY=${API_KEY_ENV},BASE_MODEL_ID=${BASE_MODEL_ID}"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  deploy_cmd+=(--set-env-vars="HF_TOKEN=${HF_TOKEN}")
else
  echo "WARNING: HF_TOKEN not set. Model downloads will fail for gated Gemma weights." >&2
fi

echo ":: Deploying to Cloud Run".
"${deploy_cmd[@]}"

echo ":: Deployment complete"
