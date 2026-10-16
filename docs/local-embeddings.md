# A local embedding server for tsukkomi

tsukkomi's dense and hybrid search call an OpenAI-compatible `POST <base url>/embeddings` endpoint. OpenRouter is the default, but any server that answers that call works (README, "Dense backends", "Embedding endpoint"). This page shows one way to run a model on your own machine. **tsukkomi ships no model weights and no server**: you download the model and run the server yourself, and tsukkomi only needs its address.

## What was tried

On one Linux machine (WSL2) with a 12 GB NVIDIA GPU, the model `Qwen/Qwen3-Embedding-0.6B` (multilingual, 1024-dimensional, about 1.2 GB of bf16 safetensors) served by the 50-line script below, through the unchanged client:

- 13 synthetic Korean and English chunks synced and embedded in 0.4 s; `audit_status` reported `dense_count` 13 and `embedder_degraded` false.
- `dense` and `hybrid` search returned the semantically matching Korean chunk first for 8 of 8 hand-written queries, among them queries that share no word with the chunk (`회의` found a chunk about a `미팅` schedule, where `sparse` found nothing).
- That is a smoke run on twelve sentences, not a benchmark. It says nothing about the quality of this model on your sessions, about larger models, or about speed on a large index (embedding hundreds of thousands of events takes real time, and a change of model, dimension or base URL embeds every event again).

## Steps

1. Download the weights from Hugging Face (safetensors only; check the SHA-256 the model page lists for the large file). A rate-limited, resumable download that leaves the rest of the machine alone:
   `curl -fL -C - --limit-rate 3M -o model.safetensors https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/<revision>/model.safetensors` (plus `config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`).
2. Start a server on a loopback address. This reference script (needs `torch` and `transformers` 4.51 or newer; they are not tsukkomi dependencies) implements the call with last-token pooling, left padding and L2 normalization as the model card describes, and honours `dimensions` by truncating and renormalizing (the model supports Matryoshka sizes). llama.cpp's server with `--embedding`, vLLM or text-embeddings-inference serve the same endpoint; they were not tried here.
3. Point tsukkomi at it and keep the local-only switch off:

```
SELF_DIRECT_EMBEDDING_BASE_URL=http://127.0.0.1:18080/v1
SELF_DIRECT_EMBEDDING_MODEL=Qwen3-Embedding-0.6B
SELF_DIRECT_EMBEDDING_DIM=1024
OPENROUTER_API_KEY=any-non-empty-string        # the local server ignores it; keep it out of the repository as usual
SELF_DIRECT_LOCAL_ONLY=false
```

The server must be reachable only by you: bind it to `127.0.0.1` (the script does), because it has no authentication.

## The reference server

```python
"""Minimal OpenAI-compatible /v1/embeddings server for Qwen3-Embedding (last-token pooling, left padding, L2 normalize).

Local development tool, loopback only. Usage: python serve_embeddings.py MODEL_DIR [PORT] [DEVICE]
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModel, AutoTokenizer

model_dir = sys.argv[1]
port = int(sys.argv[2]) if len(sys.argv) > 2 else 18080
device = sys.argv[3] if len(sys.argv) > 3 else ("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
NATIVE = model.config.hidden_size


@torch.no_grad()
def embed(texts, dim):
    out = []
    for i in range(0, len(texts), 16):
        batch = tok(texts[i:i + 16], padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
        hidden = model(**batch).last_hidden_state
        vec = hidden[:, -1]                                   # left padding: the last position is the last real token
        vec = vec[:, :dim] if dim < NATIVE else vec           # Matryoshka: truncate, then renormalize
        vec = torch.nn.functional.normalize(vec.float(), p=2, dim=1)
        out.extend(vec.cpu().tolist())
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        if self.path != "/v1/embeddings":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
        dim = int(body.get("dimensions") or NATIVE)
        data = [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(embed(texts, dim))]
        payload = json.dumps({"object": "list", "data": data, "model": body.get("model", "local")}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


print(f"serving {model_dir} on 127.0.0.1:{port} device={device} dim={NATIVE}", flush=True)
ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
```
