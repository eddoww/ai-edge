# ai-edge

Small HTTP front door for [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI).

CLIProxyAPI plugins cannot add `/v1/*` routes, so three things live here instead:

| Endpoint | Upstream | Notes |
| --- | --- | --- |
| `POST /v1/systemone` | TypeSafe `api.typesafe.ai` | Jev / System One typed questions. Defaults `model` to `jev-latest`; retries 429/529. |
| `POST /v1/embeddings` | Mistral `/v1/embeddings` | Maps OpenAI-style model names to `mistral-embed`; keeps `mistral-*` models as requested. |
| `POST /v1/audio/transcriptions` | Groq `/audio/transcriptions` | The original bytes are reused unchanged when a non-blank `model` field is present; otherwise the body is rebuilt once with the model part added, preserving existing part headers, encoding and order. Uploads are capped at 25 MiB and bounded by a concurrency semaphore. |

Everything else is reverse-proxied to `CPA_UPSTREAM`, including SSE streaming and WebSocket
upgrades (`/v1/realtime`, `/v1/live`), so clients keep a single base URL. Standard reverse-proxy
header rewriting applies to the proxied path.

## Auth

The three added endpoints require the same bearer key as CLIProxyAPI (`CPA_API_KEY`); comparison is
constant-time. Proxied requests are left untouched so CLIProxyAPI keeps enforcing its own keys.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `LISTEN_ADDR` | `:8080` | Listen address. |
| `CPA_UPSTREAM` | `http://cliproxyapi.ai-tools.svc.cluster.local:80` | CLIProxyAPI base URL. |
| `CPA_API_KEY` | required | Key accepted by the added endpoints. |
| `TYPESAFE_API_KEY` | — | Enables `/v1/systemone`; without it the route returns 503. |
| `TYPESAFE_BASE_URL` | `https://api.typesafe.ai` | TypeSafe API base. |
| `TYPESAFE_MODEL` | `jev-latest` | Model used when the request omits one. |
| `MISTRAL_API_KEY` | — | Enables `/v1/embeddings`. |
| `MISTRAL_EMBED_MODEL` | `mistral-embed` | Embedding model. |
| `GROQ_API_KEY` | — | Enables `/v1/audio/transcriptions`. |
| `GROQ_STT_MODEL` | `whisper-large-v3` | Default transcription model. |
| `MAX_CONCURRENT_TRANSCRIPTIONS` | `2` | Buffered uploads allowed at once. |

Missing provider credentials disable only the affected endpoint; the proxy keeps working. A missing
`CPA_API_KEY` is fatal because it is the key the added endpoints authenticate against.

## Build

```bash
go test ./...
docker build -t ghcr.io/eddoww/ai-edge:0.1.0 .
```
