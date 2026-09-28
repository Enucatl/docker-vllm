# vllm

OpenAI-compatible model endpoints on the GPU workstation, served via [vLLM](https://github.com/vllm-project/vllm).

## PaddleOCR-VL-1.6 document parsing

The full parsing API is at `http://10.0.40.133:8108`; Paperless uses this base URL without `/v1`. Its endpoints are `http://10.0.40.133:8108/layout-parsing` and `http://10.0.40.133:8108/health`. Port 8108 binds all workstation interfaces. The Compose project is separate from the existing OpenAI endpoints and retains the Nanonets configuration and cache for rollback. Only one OCR model should occupy the GPU at a time; the Qwen coding profile also remains an alternate GPU allocation.

```bash
# Start (from /opt/docker/vllm)
docker compose --profile extraction stop nanonets-ocr
docker compose -f compose.paddle.yaml up -d --wait --wait-timeout 900

# Health and complete-PDF smoke test, including a 41-page PDF
curl -fsS http://10.0.40.133:8108/health
uv sync --group dev
uv run python scripts/verify_paddle.py --endpoint http://10.0.40.133:8108 --timeout 600

# Shut down, retaining both pinned model directories and the vLLM cache
docker compose -f compose.paddle.yaml down

# Roll back to Nanonets OCR and embeddings
docker compose -f compose.paddle.yaml down
docker compose --profile extraction up -d nanonets-ocr bge
```

`uv run python scripts/verify_paddle.py --generate` regenerates the fixed PDFs in `examples/paddle/`. The smoke test sends each complete PDF as `file` with `fileType: 0`, layout detection enabled, all Markdown labels included, `returnMarkdownImages: false`, `visualize: false`, and `restructurePages: false`. It checks source and response page counts, first/middle/last page order, blank-page preservation, table/caption/equation and Unicode text, absence of image or export payloads, and invalid-PDF rejection. It prints elapsed time and peak GPU memory. The gateway accepts large base64 bodies; its upstream and PaddleX's recognition client allow 700 seconds. Paperless's initial timeout of 600 seconds needs no change for these samples. Set `INFERENCE_OCR_BACKEND=paddleocr`, `INFERENCE_OCR_ENDPOINT=http://10.0.40.133:8108`, `INFERENCE_PADDLE_TIMEOUT=600`, and `OCR_CONCURRENCY=1` in Paperless when its separate cutover is approved.

Tested on RTX 5090 32 GB with driver 590.48.01 (CUDA 13.1). The pinned images in `compose.paddle.yaml` are PaddleOCR API `sha256:0971c409d1cab2b12aa17b76855e36ac8eb9fb1adc97dbeea15e9b09432a4a3b`, PaddleOCR vLLM `sha256:bffd525308facf5dba2f8eca44ab476704a0ae3bfdcba25f77655973e4c0a7ca`, and Nginx `sha256:0985e772fb9f729e6fa0980da05fca5d9c468e870eed43071545afa9d2e27d94`. Installed versions: PaddleOCR 3.6.0, PaddleX 3.6.1, PaddlePaddle 3.2.1, vLLM 0.10.2, Nginx 1.30.5. Pinned model revisions in `model-cache/`: PaddleOCR-VL-1.6 `c5630abae1d940eafe0697512a0325494b02ab42`; PP-DocLayoutV3 `7b48a7566925fa464281f930c58eee04fe2c862a`.

| Fixed sample | Pages | Elapsed | Peak GPU memory | Outcome |
| --- | ---: | ---: | ---: | --- |
| Multipage text and Unicode | 3 | 0.9 s | 23,887 MiB | ordered; accents, footer retained |
| Table, caption, equation | 1 | 0.6 s | 24,789 MiB | HTML table, caption, equation retained |
| Scanned text page | 1 | 0.6 s | 27,855 MiB | raster text recognized |
| Blank middle page | 3 | 0.6 s | 24,789 MiB | empty result retained in order |
| Long document | 41 | 2.6 s | 27,849 MiB | all pages returned; first, middle, last checked |

The first table request took 42.2 seconds while the service warmed up; the first request after a pinned-image restart took 40.9 seconds. An invalid PDF returned HTTP 422. With recognition stopped, `/health` returned HTTP 500; after restart it returned 200. GPU memory remained at about 27,855 MiB after the long request, leaving about 4.6 GiB of the 32,607 MiB device memory. These measurements cover the Paddle stack alone; other GPU services were stopped during the run.

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
