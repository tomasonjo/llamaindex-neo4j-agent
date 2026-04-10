# llamaindex-neo4j-agent

A FastAPI + Streamlit app that wraps a LlamaIndex `FunctionAgent` with:

- **Retrieval Neo4j** — exposed to the agent as tools through an MCP server
  (`neo4j-mcp-server`) that the backend launches as a **stdio** child process.
  Defaults to the public `demo.neo4jlabs.com/companies2` dataset, but you can
  point it at any Neo4j instance via `.env`.
- **Memory Neo4j** — a **local** Neo4j (single database) run via
  `docker compose`, used by `neo4j-agent-memory` to give the agent persistent,
  per-session memory.

```
┌────────────┐     HTTP      ┌─────────────┐   MCP/stdio  ┌────────────────┐
│ Streamlit  │ ───────────▶  │   FastAPI   │ ──────────▶  │ neo4j-mcp      │
│    UI      │               │  (agent)    │              │  server        │──▶ Retrieval Neo4j
└────────────┘               │             │              └────────────────┘
                             │             │     Bolt     ┌────────────────┐
                             │             │ ──────────▶  │ Memory Neo4j   │ (docker, local)
                             └─────────────┘              └────────────────┘
```

## 1. Prerequisites

- Docker + Docker Compose
- Python 3.10+
- An OpenAI API key

## 2. Start the local memory Neo4j

```bash
docker compose up -d
```

Neo4j Browser: http://localhost:7474 — login `neo4j` / `password`.
Bolt: `bolt://localhost:7687`.

## 3. Install Python deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Configure environment

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

By default:
- Retrieval points at `neo4j+s://demo.neo4jlabs.com:7687` (database `companies2`).
- Memory points at the local docker Neo4j on `bolt://localhost:7687`.

## 5. Run the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

The backend:
1. Launches `neo4j-mcp-server` as a stdio child process (read-only) pointed
   at the retrieval Neo4j via `BasicMCPClient`.
2. Loads tools from it with `McpToolSpec`.
3. Opens a `MemoryClient` against the local memory Neo4j.
4. Builds a fresh `FunctionAgent` per request and injects a per-`session_id`
   `Neo4jLlamaIndexMemory`.

Endpoints:
- `GET /health`
- `POST /chat` → `{session_id, message}` → `{response, events}`
- `POST /chat/stream` → SSE stream of `delta` / `tool_call` / `tool_result` / `final`

## 6. Run the Streamlit frontend

In a second terminal:

```bash
source .venv/bin/activate
streamlit run frontend/app.py
```

Open http://localhost:8501. Each Streamlit session gets its own `session_id`
(shown in the sidebar) so the agent's memory is scoped to that conversation —
change the ID or click "New session" to start fresh.

## Notes

- The original notebook script this project is based on lives at
  `llamaindexneo4jmcp.py`.
- The agent's system prompt instructs it to always call `get-schema` before
  running Cypher. You can tweak it in `backend/agent.py`.
- To target a different retrieval Neo4j, set `RETRIEVAL_NEO4J_*` in `.env`.
