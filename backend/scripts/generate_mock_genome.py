#!/usr/bin/env python3
"""
Generate a lightweight mock FASTA genome for Cas-OFFinder integration testing.

Produces ~1 MiB of synthetic sequence data so you can exercise the real
Cas-OFFinder binary without downloading the full hg38 reference (~3 GB).

Usage (from backend/):
    python scripts/generate_mock_genome.py
    python scripts/generate_mock_genome.py --size-mb 2 --output genome/mock_genome.fa
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = BACKEND_ROOT / "genome" / "mock_genome.fa"
DEFAULT_SIZE_BYTES = 1_048_576  # 1 MiB
LINE_WIDTH = 80
BASES = "ATCG"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_mock_genome")


def _generate_sequence_chunk(length: int, seed_offset: int) -> str:
    """Deterministic pseudo-random ATCG sequence (no external deps)."""
    return "".join(BASES[(seed_offset + i) % len(BASES)] for i in range(length))


def generate_mock_fasta(
    output_path: Path,
    target_size_bytes: int,
    num_contigs: int = 5,
) -> dict[str, int]:
    """
    Write a multi-contig FASTA file approaching target_size_bytes.

    Returns summary stats: {contigs, sequence_bytes, total_bytes}.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Headers consume bytes too — target total file size, not just sequence.
    sequence_budget = max(int(target_size_bytes * 0.95), 10_000)
    per_contig = sequence_budget // num_contigs
    sequence_written = 0
    total_bytes = 0

    logger.info("Writing mock genome to %s", output_path.resolve())
    logger.info("Target sequence payload: ~%d bytes across %d contigs", sequence_budget, num_contigs)

    with output_path.open("w", encoding="utf-8") as handle:
        for contig_idx in range(1, num_contigs + 1):
            header = f">mock_chr{contig_idx} synthetic test contig for Cas-OFFinder\n"
            handle.write(header)
            total_bytes += len(header.encode("utf-8"))

            remaining = per_contig
            seed = contig_idx * 1_000
            while remaining > 0:
                chunk_len = min(LINE_WIDTH, remaining)
                line = _generate_sequence_chunk(chunk_len, seed) + "\n"
                handle.write(line)
                total_bytes += len(line.encode("utf-8"))
                sequence_written += chunk_len
                remaining -= chunk_len
                seed += chunk_len

    return {
        "contigs": num_contigs,
        "sequence_bytes": sequence_written,
        "total_bytes": total_bytes,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a mock FASTA genome for Cas-OFFinder testing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output FASTA path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--size-mb",
        type=float,
        default=1.0,
        help="Approximate target file size in megabytes (default: 1.0)",
    )
    parser.add_argument(
        "--contigs",
        type=int,
        default=5,
        help="Number of synthetic contigs (default: 5)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    output_path = args.output
    if not output_path.is_absolute():
        output_path = (BACKEND_ROOT / output_path).resolve()

    target_bytes = int(args.size_mb * 1_048_576)

    try:
        stats = generate_mock_fasta(
            output_path=output_path,
            target_size_bytes=target_bytes,
            num_contigs=max(args.contigs, 1),
        )
    except OSError as exc:
        logger.error("Failed to write mock genome: %s", exc)
        return 1

    logger.info("Done — mock genome created successfully.")
    logger.info("  Contigs         : %d", stats["contigs"])
    logger.info("  Sequence bytes  : %d", stats["sequence_bytes"])
    logger.info("  Total file size : %d bytes (%.2f MiB)", stats["total_bytes"], stats["total_bytes"] / 1_048_576)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Set CAS_OFFINDER_GENOME_PATH=./genome/mock_genome.fa in backend/.env")
    logger.info("  2. Place cas-offinder.exe in backend/bin/")
    logger.info("  3. Restart the FastAPI server and run an evaluation.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
