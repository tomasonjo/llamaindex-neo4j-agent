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

## 2. Configure environment

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

By default:
- Retrieval points at `neo4j+s://demo.neo4jlabs.com:7687` (database `companies2`).
- Memory points at the dockerized Neo4j (`bolt://neo4j-memory:7687` from
  inside the compose network, `bolt://localhost:7687` from your host).

## 3. Run everything with Docker Compose

```bash
docker compose up -d --build
```

This starts three services:
- **neo4j-memory** — latest Neo4j image with APOC, ports `7474`/`7687`
  (`neo4j` / `password`) — used for agent memory.
- **backend** — FastAPI + LlamaIndex agent on `http://localhost:8000`.
  Spawns `neo4j-mcp-server` as a stdio child process on startup.
- **frontend** — Streamlit UI on `http://localhost:8501`, pointed at
  the backend service via `BACKEND_URL=http://backend:8000`.

Open the UI at **http://localhost:8501**.

### Backend details

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

### Running locally without Docker (optional)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d neo4j-memory                # only the memory Neo4j
uvicorn backend.main:app --reload --port 8000    # backend
streamlit run frontend/app.py                    # frontend (another terminal)
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
