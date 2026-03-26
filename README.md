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

## Example queries

### OCR — Nanonets-OCR2-3B (port 8100)

The vision model needs a base64-encoded image (external URLs get blocked by most hosts).
A test image is included at `examples/test-ocr.png`.

```bash
IMG_B64=$(base64 -w0 < examples/test-ocr.png)

curl -s http://complex.home.arpa:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"nanonets/Nanonets-OCR2-3B\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": [
          {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,\${IMG_B64}\"}},
          {\"type\": \"text\", \"text\": \"Extract all text from this image.\"}
        ]
      }
    ],
    \"max_tokens\": 1024
  }" | jq .choices[0].message.content
```

### Embeddings — bge-m3 (port 8102)

```bash
curl -s http://complex.home.arpa:8102/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-m3",
    "input": "Hello, world!"
  }' | jq '{model: .model, dimensions: (.data[0].embedding | length)}'
```

### Structured extraction — NuExtract-2.0-4B (port 8103)

Template values specify the expected type (`string`, `date-time`, `["string"]` for arrays, etc).
The chat template wraps the schema and document as `# Template:` / `# Context:` sections.

```bash
curl -s http://complex.home.arpa:8103/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "numind/NuExtract-2.0-4B",
    "messages": [
      {
        "role": "user",
        "content": "# Template:\n{\"name\": \"string\", \"company\": \"string\", \"role\": \"string\", \"location\": \"string\"}\n# Context:\nJohn Smith works at Acme Corp as a software engineer in New York."
      }
    ],
    "max_tokens": 256
  }' | jq .choices[0].message.content
```

Models are downloaded on first start and cached in the `hf-cache` volume.

## Requirements

- NVIDIA GPU with drivers installed
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
