# Mem0 Memory Provider

Server-side LLM fact extraction with semantic search, reranking, and automatic deduplication.

## Requirements

- `pip install mem0ai`

### Cloud Mode
- Mem0 API key from [app.mem0.ai](https://app.mem0.ai)

### Local Mode
- Ollama running locally
- Embedding model (default: `qwen3-embedding:4b`)

## Setup

```bash
hermes memory setup    # select "mem0"
```

Or manually:

### Cloud Mode
```bash
hermes config set memory.provider mem0
echo "MEM0_API_KEY=your-key" >> ~/.hermes/.env
```

### Local Mode
```bash
hermes config set memory.provider mem0
cat > ~/.hermes/mem0.json << EOF
{
  "mode": "local",
  "local": {
    "llm_provider": "hermes",
    "embedding": {
      "provider": "ollama",
      "model": "qwen3-embedding:4b",
      "ollama_base_url": "http://localhost:11434"
    },
    "vector_store": {
      "provider": "qdrant",
      "path": "~/.hermes/qdrant"
    }
  }
}
EOF
```

## Config

Config file: `$HERMES_HOME/mem0.json`

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `cloud` | Connection mode: `cloud` or `local` |
| `user_id` | `hermes-user` | User identifier on Mem0 |
| `agent_id` | `hermes` | Agent identifier |
| `rerank` | `true` | Enable reranking for recall |

### Cloud Mode

| Key | Default | Description |
|-----|---------|-------------|
| `api_key` | - | Mem0 Platform API key (required) |

### Local Mode

| Key | Default | Description |
|-----|---------|-------------|
| `local.llm_provider` | `hermes` | LLM provider (use `hermes` to reuse hermes config) |
| `local.embedding.provider` | `ollama` | Embedding provider |
| `local.embedding.model` | `qwen3-embedding:4b` | Embedding model |
| `local.embedding.base_url` | `http://localhost:11434` | Ollama base URL |
| `local.vector_store.provider` | `qdrant` | Vector store provider |
| `local.vector_store.path` | `~/.hermes/qdrant` | Vector store path |

## Tools

| Tool | Description |
|------|-------------|
| `mem0_profile` | All stored memories about the user |
| `mem0_search` | Semantic search with optional reranking |
| `mem0_conclude` | Store a fact verbatim (no LLM extraction) |
