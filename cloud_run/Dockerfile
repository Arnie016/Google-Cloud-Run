FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_NO_TORCHAO=1 \
    TOKENIZERS_PARALLELISM=false \
    BITSANDBYTES_NOWELCOME=1 \
    DISABLE_BITSANDBYTES_AUTO_INSTALL=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

COPY cloud_run/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app

COPY cloud_run/app /app/app
COPY models/gemma3-legal-lora-original.tar.gz /tmp/lora.tar.gz
RUN mkdir -p /app/models/lora \
    && tar -xzf /tmp/lora.tar.gz -C /app/models/lora --strip-components=1 \
    && rm /tmp/lora.tar.gz

ENV BASE_MODEL_ID=google/gemma-3-4b-it \
    LORA_PATH=/app/models/lora \
    API_KEY=change-me

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
