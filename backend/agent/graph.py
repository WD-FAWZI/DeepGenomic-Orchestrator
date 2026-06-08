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
from agent.tools import run_cas_offinder, score_with_hyenadna, run_biological_filters

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
    biological_filter_result: dict[str, Any] | None
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


def biological_filter_node(state: GraphState) -> GraphState:
    """Run biophysical filters against the derived guide sequence."""
    guide = state.get("guide_sequence", "")
    result = run_biological_filters(guide)

    return {
        **state,
        "current_step": "biological_filter_complete",
        "biological_filter_result": result,
    }


def route_after_filter(state: GraphState) -> str:
    """Determine next step based on biological viability (short-circuit)."""
    result = state.get("biological_filter_result")
    if result and result.get("viable") is True:
        return "cas_offinder"
    return "synthesize"


def cas_offinder_node(state: GraphState) -> GraphState:
    """Run the Cas-OFFinder tool against the derived guide sequence."""
    guide = state.get("guide_sequence", "")
    result = run_cas_offinder(guide)

    return {
        **state,
        "current_step": "cas_offinder_complete",
        "cas_offinder_result": result,
    }


async def hyenadna_node(state: GraphState) -> GraphState:
    """Score the target sequence with the HyenaDNA model, injecting flanking context if short."""
    target = state.get("input_sequence", "")
    metadata = dict(state.get("metadata", {}))

    extracted_seq = None
    if len(target) < 100:
        cas_result = state.get("cas_offinder_result")
        on_target = None
        if cas_result and isinstance(cas_result, dict):
            off_targets = cas_result.get("off_targets", [])
            for ot in off_targets:
                if ot.get("mismatches") == 0:
                    on_target = ot
                    break

        if on_target:
            chrom = on_target.get("chromosome")
            pos = on_target.get("position")
            strand = on_target.get("strand", "+")
            if chrom is not None and pos is not None:
                from agent.tools import extract_flanking_context
                genome_path = cas_result.get("genome_path")
                extracted_seq = extract_flanking_context(
                    chromosome=chrom,
                    position=pos,
                    strand=strand,
                    genome_fasta_path=genome_path,
                )
                if extracted_seq:
                    metadata["extracted_context"] = {
                        "chromosome": chrom,
                        "position": pos,
                        "strand": strand,
                        "length": len(extracted_seq),
                    }

    eval_sequence = extracted_seq if extracted_seq else target
    result = await score_with_hyenadna(eval_sequence)

    return {
        **state,
        "current_step": "hyenadna_complete",
        "hyenadna_score": result,
        "metadata": metadata,
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
    filter_result: dict[str, Any] | None = None,
) -> str:
    """Construct the bioinformatics assessment prompt for the local LLM."""
    filter_text = ""
    rejection_instruction = ""
    if filter_result:
        viable = filter_result.get("viable", False)
        filter_text = (
            f"Biological Filters Report:\n"
            f"- Viable: {viable}\n"
            f"- Biophysical Score: {filter_result.get('score')}\n"
            f"- GC Content: {filter_result.get('gc_content')}\n"
            f"- Shannon Entropy: {filter_result.get('shannon_entropy')}\n"
            f"- Homopolymer Run Detected: {filter_result.get('has_homopolymer')}\n"
            f"- Poly-T (U6 promoter termination) Risk: {filter_result.get('has_polyT_u6')}\n"
            f"- Max Self-Complementarity Match: {filter_result.get('self_comp_max')} bp\n"
            f"- Filter reasons/notes: {filter_result.get('reasons')}\n\n"
        )
        if not viable:
            rejection_instruction = (
                "CRITICAL: The sequence has failed biophysical validation filters and has been biologically "
                "rejected before deep genomic scanning occurred. You MUST explicitly state that the sequence "
                "was biologically rejected due to these biophysical issues, explaining them clearly.\n\n"
            )

    hyena_text = (
        f"{hyena_result:.4f} ({hyena_result:.1%} predicted efficiency)"
        if hyena_result is not None
        else "Unavailable (Bypassed due to rejection or model not loaded)"
    )

    cas_text = (
        _format_tool_result(cas_result)
        if cas_result is not None
        else "Unavailable (Bypassed due to rejection)"
    )

    return (
        "You are an expert bioinformatician. Based on the following biological filters, off-target "
        "alignment, and scoring data, write a brief, 2-sentence professional assessment of this "
        "CRISPR guide's viability.\n\n"
        f"{rejection_instruction}"
        f"{filter_text}"
        f"Cas-OFFinder (off-target analysis):\n{cas_text}\n\n"
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
    cas_result = state.get("cas_offinder_result")
    hyena_result = state.get("hyenadna_score")
    filter_result = state.get("biological_filter_result")
    guide_sequence = state.get("guide_sequence", "")
    metadata = dict(state.get("metadata", {}))

    prompt = _build_synthesis_prompt(
        cas_result=cas_result,
        hyena_result=hyena_result,
        guide_sequence=guide_sequence,
        filter_result=filter_result,
    )

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
            else ("bypassed (rejected)" if filter_result and not filter_result.get("viable") else "unavailable")
        )
        cas_fallback = (
            _format_tool_result(cas_result)
            if cas_result is not None
            else ("bypassed (rejected)" if filter_result and not filter_result.get("viable") else "unavailable")
        )
        filter_fallback = ""
        if filter_result:
            filter_fallback = (
                f"- Biophysical viability: {filter_result.get('viable')} "
                f"(Score: {filter_result.get('score')}, Reasons: {filter_result.get('reasons')})\n"
            )

        final_evaluation = (
            f"[Ollama unavailable at {OLLAMA_BASE_URL}: {exc}]\n\n"
            "Structured fallback assessment:\n"
            f"{filter_fallback}"
            f"- Off-target analysis: {cas_fallback}\n"
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
    graph.add_node("biological_filter", biological_filter_node)
    graph.add_node("cas_offinder", cas_offinder_node)
    graph.add_node("hyenadna", hyenadna_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("init")
    graph.add_edge("init", "biological_filter")

    # Conditional routing after filter checks
    graph.add_conditional_edges(
        "biological_filter",
        route_after_filter,
        {
            "cas_offinder": "cas_offinder",
            "synthesize": "synthesize",
        }
    )

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


async def run_evaluation(
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
        "biological_filter_result": None,
        "final_evaluation": None,
        "metadata": {},
    }

    graph = _get_graph()
    result: GraphState = await graph.ainvoke(initial)
    return AgentState.from_graph_dict(dict(result))

