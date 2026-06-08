"""
LangGraph evaluation pipeline.

Flow:
  init → cas_offinder → hyenadna_score → synthesize → END

The synthesize node delegates the final assessment to a local Ollama instance
via LangChain's Ollama LLM wrapper (langchain_community.llms).
"""

from __future__ import annotations

import json
import os
from typing import Any, TypedDict

from langchain_community.llms import Ollama
from langgraph.graph import END, StateGraph

from agent.state import AgentState
from agent.tools import run_cas_offinder, score_with_hyenadna

# ---------------------------------------------------------------------------
# Ollama configuration — override via environment variables if needed.
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:latest")

_ollama_llm: Ollama | None = None

# ---------------------------------------------------------------------------
# LangGraph state schema (TypedDict is required by LangGraph's StateGraph).
# Fields mirror AgentState in state.py — keep them in sync.
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    input_sequence: str
    current_step: str
    guide_sequence: str
    cas_offinder_result: dict[str, Any] | None
    hyenadna_score: float | None
    final_evaluation: str | None
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def _derive_guide_sequence(dna_sequence: str) -> str:
    """Derive a placeholder guide sequence from the first 20 bases."""
    cleaned = dna_sequence.upper().strip()
    return cleaned[:20] if cleaned else ""


def init_node(state: GraphState) -> GraphState:
    """Normalize input and prepare derived fields."""
    sequence = state.get("input_sequence", "").upper().strip()
    guide = state.get("guide_sequence") or _derive_guide_sequence(sequence)

    return {
        **state,
        "input_sequence": sequence,
        "guide_sequence": guide,
        "current_step": "initialized",
        "metadata": {
            **state.get("metadata", {}),
            "pipeline": "deepgenomic-eval-v0",
        },
    }


def cas_offinder_node(state: GraphState) -> GraphState:
    """Run the Cas-OFFinder tool against the derived guide sequence."""
    guide = state.get("guide_sequence", "")
    result = run_cas_offinder(guide)

    return {
        **state,
        "current_step": "cas_offinder_complete",
        "cas_offinder_result": result,
    }


def hyenadna_node(state: GraphState) -> GraphState:
    """Score the full target sequence with the HyenaDNA model."""
    target = state.get("input_sequence", "")
    result = score_with_hyenadna(target)

    return {
        **state,
        "current_step": "hyenadna_complete",
        "hyenadna_score": result,
    }


def _get_ollama_llm() -> Ollama:
    """Return a lazily-initialized Ollama LLM connected to the local instance."""
    global _ollama_llm
    if _ollama_llm is None:
        _ollama_llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    return _ollama_llm


def _format_tool_result(result: Any) -> str:
    """Serialize tool output (dict or str) into a prompt-friendly string."""
    if isinstance(result, dict):
        return json.dumps(result, indent=2)
    return str(result)


def _build_synthesis_prompt(
    cas_result: Any,
    hyena_result: float | None,
    guide_sequence: str,
) -> str:
    """Construct the bioinformatics assessment prompt for the local LLM."""
    hyena_text = (
        f"{hyena_result:.4f} ({hyena_result:.1%} predicted efficiency)"
        if hyena_result is not None
        else "Unavailable (HyenaDNA model not loaded or inference failed)"
    )
    return (
        "You are an expert bioinformatician. Based on the following off-target "
        "alignment and scoring data, write a brief, 2-sentence professional "
        "assessment of this CRISPR guide's viability.\n\n"
        f"Cas-OFFinder (off-target analysis):\n{_format_tool_result(cas_result)}\n\n"
        f"HyenaDNA (efficiency scoring):\n{hyena_text}\n\n"
        f"Guide sequence: {guide_sequence}"
    )


def synthesize_node(state: GraphState) -> GraphState:
    """
    Invoke the local Ollama LLM to produce a professional viability assessment.

    Reads tool outputs from state, sends them to the local LLM, and stores the
    generated response as final_evaluation. Falls back to a structured summary
    if Ollama is unreachable so the pipeline never hard-fails.
    """
    cas_result = state.get("cas_offinder_result") or "No off-target data available."
    hyena_result = state.get("hyenadna_score")
    guide_sequence = state.get("guide_sequence", "")
    metadata = dict(state.get("metadata", {}))

    prompt = _build_synthesis_prompt(cas_result, hyena_result, guide_sequence)

    try:
        llm = _get_ollama_llm()
        final_evaluation = llm.invoke(prompt).strip()
        metadata["llm_provider"] = "ollama"
        metadata["llm_model"] = OLLAMA_MODEL
        metadata["llm_base_url"] = OLLAMA_BASE_URL
    except Exception as exc:
        # Graceful fallback — keep the API responsive when Ollama is offline.
        hyena_fallback = (
            f"{hyena_result:.4f} ({hyena_result:.1%})"
            if hyena_result is not None
            else "unavailable"
        )
        final_evaluation = (
            f"[Ollama unavailable at {OLLAMA_BASE_URL}: {exc}]\n\n"
            "Structured fallback assessment:\n"
            f"- Off-target analysis: {_format_tool_result(cas_result)}\n"
            f"- Efficiency scoring: {hyena_fallback}\n"
            f"- Guide sequence ({len(guide_sequence)} nt): {guide_sequence}"
        )
        metadata["llm_error"] = str(exc)

    return {
        **state,
        "current_step": "complete",
        "final_evaluation": final_evaluation,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Graph builder & runner
# ---------------------------------------------------------------------------


def build_evaluation_graph():
    """Construct and compile the evaluation StateGraph."""
    graph = StateGraph(GraphState)

    graph.add_node("init", init_node)
    graph.add_node("cas_offinder", cas_offinder_node)
    graph.add_node("hyenadna", hyenadna_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("init")
    graph.add_edge("init", "cas_offinder")
    graph.add_edge("cas_offinder", "hyenadna")
    graph.add_edge("hyenadna", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


# Module-level compiled graph (reused across requests).
_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_evaluation_graph()
    return _compiled_graph


def run_evaluation(
    dna_sequence: str,
    guide_sequence: str | None = None,
) -> AgentState:
    """
    Execute the full evaluation pipeline and return validated AgentState.

    Args:
        dna_sequence: Raw DNA sequence from the API caller.
        guide_sequence: Optional override for the derived guide sequence.

    Returns:
        AgentState populated with tool results and final evaluation.
    """
    initial: GraphState = {
        "input_sequence": dna_sequence,
        "guide_sequence": guide_sequence or "",
        "current_step": "init",
        "cas_offinder_result": None,
        "hyenadna_score": None,
        "final_evaluation": None,
        "metadata": {},
    }

    graph = _get_graph()
    result: GraphState = graph.invoke(initial)
    return AgentState.from_graph_dict(dict(result))
