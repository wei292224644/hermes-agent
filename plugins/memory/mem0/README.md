# Mem0 Memory Provider

Server-side LLM fact extraction with semantic search, reranking, and automatic deduplication.

Supports [Mem0 Cloud](https://app.mem0.ai), self-hosted instances, and a fully
local in-process mode.

## Requirements

- `pip install mem0ai`
- Mem0 Cloud API key **or** a self-hosted Mem0 server

### Local Mode
- Ollama running locally
- Embedding model (default: `qwen3-embedding:4b`)

## Setup

### Cloud

```bash
hermes memory setup    # select "mem0"
```

Or manually:

```bash
hermes config set memory.provider mem0
echo "MEM0_API_KEY=your-key" >> ~/.hermes/.env
```

### Self-Hosted

```bash
hermes config set memory.provider mem0
echo "MEM0_HOST=http://your-mem0-server:24220" >> ~/.hermes/.env
echo "MEM0_API_KEY=your-api-key" >> ~/.hermes/.env   # if auth is enabled
```

### Local Mode

```bash
hermes config set memory.provider mem0
cat > ~/.hermes/mem0.json << EOF
{
  "mode": "local",
  "llm_provider": "hermes",
  "embedding_provider": "ollama",
  "embedding_model": "qwen3-embedding:4b",
  "embedding_base_url": "http://localhost:11434",
  "vector_store_provider": "qdrant",
  "vector_store_path": "~/.hermes/qdrant"
}
EOF
```

## Config

Config file: `$HERMES_HOME/mem0.json`

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `cloud` | Connection mode: `cloud` or `local` |
| `api_key` | — | API key (required for cloud; optional for self-hosted without auth) |
| `host` | `https://api.mem0.ai` | Self-hosted Mem0 URL. When set, overrides the cloud endpoint. |
| `user_id` | `hermes-user` | User identifier |
| `agent_id` | `hermes` | Agent identifier |
| `rerank` | `true` | Enable reranking for recall |

### Cloud Mode

| Key | Default | Description |
|-----|---------|-------------|
| `api_key` | - | Mem0 Platform API key (required) |

### Local Mode

| Key | Default | Description |
|-----|---------|-------------|
| `llm_provider` | `hermes` | LLM provider. `hermes` (default) routes fact extraction through hermes's own model/credentials via `call_llm` — no API key needed here. Set a concrete provider (`openai`, `ollama`, …) only to override. |
| `embedding_provider` | `ollama` | Embedding provider |
| `embedding_model` | `qwen3-embedding:4b` | Embedding model |
| `embedding_base_url` | `http://localhost:11434` | Ollama base URL |
| `vector_store_provider` | `qdrant` | Vector store provider |
| `vector_store_path` | `~/.hermes/qdrant` | Vector store path |

## Tools

| Tool | Description |
|------|-------------|
| `mem0_profile` | All stored memories about the user |
| `mem0_search` | Semantic search with optional reranking |
| `mem0_conclude` | Store a fact verbatim (no LLM extraction) |
