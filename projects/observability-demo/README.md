# Observability demo — Phoenix tracing + Prometheus metrics

A small RAG + tool-calling agent on local Ollama models, fully traced in Arize Phoenix.
Full explanation: [notes/observability-guide.md](../../notes/observability-guide.md).

```bash
ollama pull qwen2.5:7b && ollama pull nomic-embed-text
uv sync
uv run phoenix serve          # terminal 1 → http://localhost:6006
uv run python app.py          # terminal 2 → 4 demo questions, traces appear in Phoenix
```

Optional infra stack (Prometheus, Grafana, node-exporter, persistent Phoenix): `docker compose up -d`.

| Env var | Default | Purpose |
|---|---|---|
| `MODEL` | `qwen2.5:7b` | Chat model (needs tool calling) |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | Any OpenAI-compatible endpoint |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6006` | Phoenix server |
| `METRICS_PORT` | `8001` | Prometheus `/metrics` port |
