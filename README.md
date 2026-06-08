# DeepGenomic Orchestrator

Local-first, privacy-focused AI agent system for bioinformatics. This scaffold wires together a **Python/LangGraph backend** and a **Next.js frontend** — the "pipes" are ready; real biological tooling comes later.

## Architecture

```
frontend/          Next.js (App Router) — cyberpunk UI
  └── lib/api.ts   → POST http://localhost:8000/api/evaluate

backend/           FastAPI + LangGraph orchestration
  └── agent/
        state.py   Pydantic models (API + graph schema)
        tools.py   Mock Cas-OFFinder & HyenaDNA stubs
        graph.py   LangGraph state machine
```

### Pipeline flow

```
init → cas_offinder → hyenadna_score → synthesize_evaluation → END
```

The synthesize node calls a local Ollama instance (`llama3` by default) via LangChain.

## Quick start

### Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Verify: [http://localhost:8000/health](http://localhost:8000/health)

### Frontend

```bash
cd frontend
cp .env.local.example .env.local   # optional — defaults to localhost:8000
npm install
npm run dev
```

Open: [http://localhost:3000](http://localhost:3000)

## API

### `POST /api/evaluate`

```json
{
  "dna_sequence": "ATCGATCGATCG...",
  "guide_sequence": "optional override"
}
```

Response includes `cas_offinder_result`, `hyenadna_score`, and `final_evaluation`.

## Environment variables

| Variable | Service | Default |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Frontend | `http://localhost:8000` |
| `CORS_ORIGINS` | Backend | `http://localhost:3000` |
| `OLLAMA_MODEL` | Backend | `gemma4:latest` |
| `OLLAMA_BASE_URL` | Backend | `http://localhost:11434` |
| `CAS_OFFINDER_GENOME_PATH` | Backend | *(required for real runs)* |
| `CAS_OFFINDER_BINARY` | Backend | `cas-offinder` |
| `CAS_OFFINDER_DEVICE` | Backend | `C` (CPU) |
| `CAS_OFFINDER_PAM_PATTERN` | Backend | SpCas9 `NNNNNNNNNNNNNNNNNNNNNNRG` |

## Next steps

- Replace mock tools in `backend/agent/tools.py` with real integrations
- Enable Ollama for LLM-powered synthesis in `graph.py`
- Add authentication and sequence validation rules
- Containerize with Docker Compose for one-command local dev
