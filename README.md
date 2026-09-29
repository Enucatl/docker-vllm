# vllm

OpenAI-compatible model endpoints on the GPU workstation, served via [vLLM](https://github.com/vllm-project/vllm).

## PaddleOCR-VL-1.6 document parsing

The full parsing API is at `http://10.0.40.133:8108`; Paperless uses this base URL without `/v1`. Its endpoints are `/layout-parsing`, `/health`, and `/metadata`. Parsing responses pass through unchanged, without gateway-injected `result.provenance`. Port 8108 binds all workstation interfaces. PaddleOCR replaces the retired Nanonets OCR service in the `extraction` profile. The Qwen coding profile remains an alternate GPU allocation.

`GET /metadata` returns HTTP 200 with `Content-Type: application/json`, `Cache-Control: no-store`, and this flat JSON object:

```json
{"pipeline":"PaddleOCR-VL-1.6","model":"PaddlePaddle/PaddleOCR-VL-1.6","layout_model":"PP-DocLayoutV3"}
```

Paperless AI should fetch and validate these nonempty string fields at the start of each processing batch (or standalone document job), then persist that snapshot as provenance alongside each document's results. Metadata fetch or validation failure should fail the job before document writes; do not reuse stale metadata or invent defaults. This describes the configured deployment, not per-request attestation. Stop processing during backend upgrades so one batch cannot span deployments. `/metadata` is static and does not indicate readiness; use `/health` for that. If the pipeline or models change, update the names in `paddle-nginx.conf` and rerun the smoke test.

```bash
# One time: stop PaddleOCR under its former standalone Compose project
docker compose -f compose.paddle.yaml down

# Start PaddleOCR and BGE embeddings under the extraction profile
docker compose --profile extraction up -d --wait --wait-timeout 900

# Health and complete-PDF smoke test, including a 41-page PDF
curl -fsS http://10.0.40.133:8108/health
curl -fsS http://10.0.40.133:8108/metadata
uv sync --group dev
uv run python scripts/verify_paddle.py --endpoint http://10.0.40.133:8108 --timeout 600

# Stop extraction services before allocating the GPU to the coding profile
docker compose --profile extraction down
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
| 8108 | PaddleOCR-VL-1.6 | OCR document parsing | about 86% |
| 8102 | `BAAI/bge-m3` | Embeddings (multilingual) | 5% |
| 8107 | `Qwen/Qwen3.5-9B` | Coding LLM (coding profile) | 85% |


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

### Embeddings — bge-m3 (port 8102)

```bash
curl -s http://complex.home.arpa:8102/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-m3",
    "input": "Hello, world!"
  }' | jq '{model: .model, dimensions: (.data[0].embedding | length)}'
```

Models are downloaded on first start and cached in the `hf-cache` volume.

## Requirements

- NVIDIA GPU with drivers installed
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
