"""
Bioinformatics tool integrations for the LangGraph agent pipeline.

Each function is a thin adapter between the orchestration layer and the
underlying compute backends in bio_core/.
"""

from __future__ import annotations

import os
import asyncio
from typing import Any

from config.environment import load_environment, resolve_env_path
from bio_core.cas_wrapper import (
    CasOffinderConfigError,
    CasOffinderError,
    CasOffinderRunner,
)
from bio_core.hyena_inference import HyenaZeroShotEvaluator, HyenaDNAInferenceError
from bio_core.filters import evaluate_guide

# Ensure .env is loaded when tools are invoked outside the FastAPI process.
load_environment()

_hyena_evaluator: HyenaZeroShotEvaluator | None = None


def run_biological_filters(guide_seq: str) -> dict[str, Any]:
    """
    Evaluate the biological viability of the guide sequence using biophysical rules.

    Args:
        guide_seq: The 20bp guide sequence to evaluate.

    Returns:
        Dictionary containing the GuideFilterReport attributes.
    """
    report = evaluate_guide(guide_seq)
    return report.to_dict()


def run_cas_offinder(
    guide_seq: str,
    genome_fasta_path: str | None = None,
    mismatch_threshold: int = 3,
) -> dict[str, Any]:
    """
    Run a real Cas-OFFinder off-target search against a local genome.

    Requires local genome FASTA and cas-offinder binary.

    Configuration (environment variables):
        CAS_OFFINDER_GENOME_PATH  — path to genome FASTA file or directory
        CAS_OFFINDER_BINARY       — path to cas-offinder executable (default: cas-offinder)
        CAS_OFFINDER_DEVICE       — G | C | A  (default: C for CPU)
        CAS_OFFINDER_PAM_PATTERN  — PAM pattern line (default: SpCas9 20bp + NRG)

    Args:
        guide_seq: Guide RNA / protospacer sequence to search.
        genome_fasta_path: Optional override for the genome path (else env var).
        mismatch_threshold: Maximum mismatches to report (default: 3).

    Returns:
        Dictionary with parsed off-target hits and run metadata, e.g.:
        {
            "off_targets": [{"chromosome": "chr1", "position": 12345, ...}],
            "hit_count": 1,
            "guide_sequence": "ATCG...",
            "genome_path": "/path/to/genome",
            "mismatch_threshold": 3,
        }

        On failure, returns the same shape with "error" populated and an empty
        off_targets list so the orchestration pipeline can continue gracefully.
    """
    guide = guide_seq.strip()
    if not guide:
        return _cas_error_result(
            guide_sequence="",
            error="No guide sequence provided — skipping off-target scan.",
        )

    genome_raw = genome_fasta_path or os.getenv("CAS_OFFINDER_GENOME_PATH")
    if not genome_raw:
        return _cas_error_result(
            guide_sequence=guide,
            error=(
                "CAS_OFFINDER_GENOME_PATH is not configured. "
                "Set the environment variable to a local genome FASTA file or directory."
            ),
        )

    genome_path = str(resolve_env_path(genome_raw))
    binary_raw = os.getenv("CAS_OFFINDER_BINARY", "cas-offinder")
    binary_path = str(resolve_env_path(binary_raw, default="./bin/cas-offinder.exe"))

    try:
        runner = CasOffinderRunner(
            binary_path=binary_path,
            device=os.getenv("CAS_OFFINDER_DEVICE", "C"),
            pam_pattern=os.getenv(
                "CAS_OFFINDER_PAM_PATTERN",
                CasOffinderRunner.DEFAULT_PAM_PATTERN,
            ),
        )
        result = runner.run(
            guide_sequence=guide,
            genome_fasta_path=genome_path,
            mismatch_threshold=mismatch_threshold,
        )
        return result.to_dict()

    except CasOffinderConfigError as exc:
        return _cas_error_result(guide_sequence=guide, error=f"Configuration error: {exc}")
    except CasOffinderError as exc:
        return _cas_error_result(guide_sequence=guide, error=f"Cas-OFFinder failed: {exc}")
    except Exception as exc:
        return _cas_error_result(
            guide_sequence=guide,
            error=f"Unexpected error during Cas-OFFinder run: {exc}",
        )


def _cas_error_result(guide_sequence: str, error: str) -> dict[str, Any]:
    """Build a consistent error envelope that mirrors a successful result shape."""
    return {
        "off_targets": [],
        "hit_count": 0,
        "guide_sequence": guide_sequence,
        "genome_path": os.getenv("CAS_OFFINDER_GENOME_PATH", "") or "",
        "mismatch_threshold": 3,
        "error": error,
    }


def warmup_hyenadna_model() -> tuple[bool, str]:
    """
    Preload HyenaDNA into memory (preferably GPU) for low-latency requests.
    """
    global _hyena_evaluator

    if _hyena_evaluator is not None and _hyena_evaluator.model is not None:
        return True, "HyenaDNA model already loaded."

    model_raw = os.getenv("HYENA_MODEL_PATH", "").strip()
    # Support both local paths and HF repo names
    if not model_raw:
        model_name_or_path = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
    else:
        resolved_path = resolve_env_path(model_raw)
        # If it exists locally, use resolved path, otherwise try loading as repo ID directly
        if resolved_path.exists():
            model_name_or_path = resolved_path
        else:
            model_name_or_path = model_raw

    try:
        _hyena_evaluator = HyenaZeroShotEvaluator(
            model_path=model_name_or_path,
            device=os.getenv("HYENA_DEVICE", "").strip() or None,
        )
        _hyena_evaluator.load_model()
        return True, f"HyenaDNA model loaded from {model_name_or_path}"
    except Exception as exc:
        return False, f"Failed to load HyenaDNA model: {exc}"


def _get_hyenadna_inference() -> HyenaZeroShotEvaluator:
    """
    Lazy accessor used by runtime inference if startup preload was skipped.
    """
    global _hyena_evaluator
    if _hyena_evaluator is None:
        ok, message = warmup_hyenadna_model()
        if not ok or _hyena_evaluator is None:
            raise HyenaDNAInferenceError(message)
    # Ensure it's loaded even if instantiation didn't fail but loading was deferred
    if _hyena_evaluator.model is None:
        _hyena_evaluator.load_model()
    return _hyena_evaluator


async def score_with_hyenadna(target_seq: str) -> float | None:
    """
    Run real HyenaDNA efficiency inference and return a numeric score.

    Args:
        target_seq: Full or partial DNA target sequence for model scoring.

    Returns:
        Floating-point efficiency score in [0, 1], or None if unavailable.
    """
    normalized = target_seq.upper().strip()
    if not normalized:
        return None

    try:
        evaluator = _get_hyenadna_inference()
        score = await asyncio.to_thread(evaluator.predict_efficiency, normalized)
        return score
    except Exception:
        # Preserve graceful behavior while setup/model availability evolves.
        return None


_fasta_extractor: Any = None


def extract_flanking_context(
    chromosome: str,
    position: int,
    strand: str,
    genome_fasta_path: str | None = None,
) -> str | None:
    """
    Extract a 2023bp genomic context window ([position - 1000, position + 23 + 1000])
    from the reference genome, and reverse-complement it if strand is "-".

    Args:
        chromosome: Chromosome name (e.g. chr1).
        position: 0-indexed start position of the 23bp site.
        strand: "+" or "-" indicating sequence orientation.
        genome_fasta_path: Optional override for reference genome path.

    Returns:
        The 2023bp sequence string, or None if extraction fails.
    """
    global _fasta_extractor

    genome_raw = genome_fasta_path or os.getenv("CAS_OFFINDER_GENOME_PATH")
    if not genome_raw:
        return None

    try:
        from bio_core.fasta_extractor import FastaExtractor, reverse_complement

        resolved = resolve_env_path(genome_raw)
        if _fasta_extractor is None or _fasta_extractor.genome_path != resolved:
            _fasta_extractor = FastaExtractor(resolved)

        start_pos = position - 1000
        length = 2023

        extracted = _fasta_extractor.extract_sequence(chromosome, start_pos, length)

        if strand == "-":
            extracted = reverse_complement(extracted)

        return extracted
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error(
            "Failed to extract flanking context for %s:%d (strand %s): %s",
            chromosome, position, strand, exc
        )
        return None

