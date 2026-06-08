"""
Pydantic models for LangGraph state and API request/response contracts.

LangGraph internally operates on plain dicts derived from these models.
The Pydantic layer gives us validation at API boundaries and a single
source of truth for the orchestration pipeline schema.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Canonical state carried through the LangGraph evaluation pipeline."""

    input_sequence: str = Field(
        default="",
        description="Raw DNA sequence submitted for evaluation.",
    )
    current_step: str = Field(
        default="init",
        description="Name of the active pipeline step (for observability).",
    )
    guide_sequence: str = Field(
        default="",
        description="Derived guide RNA / PAM-adjacent sequence for Cas-OFFinder.",
    )
    cas_offinder_result: dict[str, Any] | None = Field(
        default=None,
        description="Structured Cas-OFFinder off-target results (hits + metadata).",
    )
    hyenadna_score: float | None = Field(
        default=None,
        description="HyenaDNA efficiency score in [0, 1].",
    )
    final_evaluation: str | None = Field(
        default=None,
        description="Aggregated evaluation produced at the end of the graph.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional key/value bag for future LLM or tool metadata.",
    )

    def to_graph_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict LangGraph nodes can read and mutate."""
        return self.model_dump()

    @classmethod
    def from_graph_dict(cls, data: dict[str, Any]) -> "AgentState":
        """Rehydrate from LangGraph node output."""
        return cls.model_validate(data)


class EvaluateRequest(BaseModel):
    """Incoming payload for POST /api/evaluate."""

    dna_sequence: str = Field(
        ...,
        min_length=1,
        description="DNA sequence to evaluate through the orchestration pipeline.",
    )
    guide_sequence: str | None = Field(
        default=None,
        description="Optional guide sequence; auto-derived from input if omitted.",
    )


class EvaluateResponse(BaseModel):
    """Structured response returned to the frontend."""

    input_sequence: str
    current_step: str
    cas_offinder_result: dict[str, Any] | None
    hyenadna_score: float | None
    final_evaluation: str
    metadata: dict[str, Any] = Field(default_factory=dict)
