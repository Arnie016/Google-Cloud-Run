# Model Artifacts

LoRA adapter weights are not committed to this repository.

Download the adapter bundle from your artifact store (GCS or Artifact Registry) and unpack to `models/gemma3-legal-lora/`:

```
mkdir -p models/gemma3-legal-lora
tar -xzf gemma3-legal-lora-original.tar.gz -C models/gemma3-legal-lora --strip-components=1
```

Contents should include:
- `adapter_config.json`
- `adapter_model.safetensors`
- tokenizer assets (`tokenizer.json`, `special_tokens_map.json`, etc.)

Update deployment instructions (`cloud_run/deploy.sh`) with the correct download path if the bundle is hosted in a bucket.
