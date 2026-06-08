"""
DeepGenomic-Orchestrator — FastAPI entry point.

Run locally:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent.graph import run_evaluation
from agent.tools import warmup_hyenadna_model
from agent.state import EvaluateRequest, EvaluateResponse
from config.environment import (
    check_cas_offinder_setup,
    load_environment,
    print_setup_warning,
)

# Load .env before any route handlers or startup checks read os.environ.
load_environment()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DeepGenomic Orchestrator",
    description="Local-first bioinformatics agent orchestration API",
    version="0.1.0",
)

# Allow the Next.js dev server to call this API directly.
# Default to wildcard for local dev (Next.js may bind to 3001 if 3000 is taken).
# Set CORS_ORIGINS to a comma-separated list in production.
_cors_origins_raw = os.getenv("CORS_ORIGINS", "*").strip()

if _cors_origins_raw == "*":
    _cors_origins = ["*"]
    _allow_credentials = False  # required when using wildcard origins
else:
    _cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
    _allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup — validate Cas-OFFinder environment (non-blocking)
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def validate_cas_offinder_environment() -> None:
    """
    Check for Cas-OFFinder binary and genome FASTA on startup.

    Missing assets print a console warning only — the server always starts.
    """
    status = check_cas_offinder_setup()
    print_setup_warning(status)
    hyena_ok, hyena_message = warmup_hyenadna_model()
    if hyena_ok:
        try:
            print(f"\033[92m✔ [HYENA READY]\033[0m {hyena_message}")
        except UnicodeEncodeError:
            print(f"[HYENA READY] {hyena_message}")
    else:
        try:
            print(f"\033[93m⚠️ [HYENA SETUP]\033[0m {hyena_message}")
        except UnicodeEncodeError:
            print(f"[HYENA SETUP] {hyena_message}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness probe for local dev and container orchestration."""
    return {"status": "ok", "service": "deepgenomic-orchestrator"}


@app.post("/api/evaluate", response_model=EvaluateResponse)
async def evaluate_sequence(request: EvaluateRequest) -> EvaluateResponse:
    """
    Run the LangGraph evaluation pipeline on a DNA sequence.

    This endpoint is the primary integration surface for the Next.js frontend.
    """
    sequence = request.dna_sequence.strip()
    if not sequence:
        raise HTTPException(status_code=400, detail="dna_sequence must not be empty.")

    # Basic sanity check — real validation (IUPAC codes, length limits) comes later.
    if not all(base in "ATCGNatcgn" for base in sequence):
        raise HTTPException(
            status_code=400,
            detail="dna_sequence contains invalid characters. Allowed: A, T, C, G, N.",
        )

    try:
        result = run_evaluation(
            dna_sequence=sequence,
            guide_sequence=request.guide_sequence,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {exc}",
        ) from exc

    return EvaluateResponse(
        input_sequence=result.input_sequence,
        current_step=result.current_step,
        cas_offinder_result=result.cas_offinder_result,
        hyenadna_score=result.hyenadna_score,
        final_evaluation=result.final_evaluation or "Evaluation produced no output.",
        metadata=result.metadata,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
