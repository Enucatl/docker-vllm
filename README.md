# vllm

OpenAI-compatible model endpoints on the GPU workstation, served via [vLLM](https://github.com/vllm-project/vllm).

## Endpoints

| Port | Model | Task | GPU memory |
|------|-------|------|-----------|
| 8100 | `nanonets/Nanonets-OCR2-3B` | OCR (vision) | 38% |
| 8102 | `BAAI/bge-m3` | Embeddings (multilingual) | 5% |
| 8103 | `numind/NuExtract-2.0-4B` | Structured extraction | 30% |

Models start sequentially (each waits for the previous to be healthy) to avoid OOM during warm-up.

`ipc: host` is set on all containers for shared-memory performance.

The `vllm-cache` volume persists torch.compile and FlashInfer JIT cache across restarts, avoiding a ~5 min recompile on each boot.

## Setup

```bash
echo -n "hf_..." > secrets/hf_token.txt
chmod 600 secrets/hf_token.txt
setfacl -m u:100000:r secrets/hf_token.txt
```

## Usage

```bash
# Start all models
docker compose up -d

# Stop
docker compose down
```

Models are downloaded on first start and cached in the `hf-cache` volume.

## Requirements

- NVIDIA GPU with drivers installed
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
