"""
Cas-OFFinder CLI wrapper.

Encapsulates input-file generation, subprocess execution, output parsing, and
temporary-file cleanup for the Cas-OFFinder off-target search binary.

Reference: https://github.com/snugel/cas-offinder
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

# Valid DNA/IUPAC characters accepted in guide sequences.
_DNA_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[ATCGNRYSWKMBDHV]+$", re.IGNORECASE)

# Cas-OFFinder output schemas (column counts vary by version).
_MODERN_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "query_id",
    "bulge_type",
    "query_sequence",
    "sequence",
    "chromosome",
    "position",
    "strand",
    "mismatches",
    "bulge_size",
)
_LEGACY_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "query_sequence",
    "chromosome",
    "position",
    "sequence",
    "strand",
    "mismatches",
)


# ---------------------------------------------------------------------------
# Exceptions — narrow, actionable error types for upstream handling.
# ---------------------------------------------------------------------------


class CasOffinderError(Exception):
    """Base exception for all Cas-OFFinder wrapper failures."""


class CasOffinderConfigError(CasOffinderError):
    """Raised when constructor or run() arguments are invalid."""


class CasOffinderBinaryNotFoundError(CasOffinderError):
    """Raised when the cas-offinder executable cannot be located."""


class CasOffinderExecutionError(CasOffinderError):
    """Raised when the cas-offinder subprocess exits with an error."""


class CasOffinderParseError(CasOffinderError):
    """Raised when the output file cannot be parsed."""


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OffTargetHit:
    """A single off-target site reported by Cas-OFFinder."""

    chromosome: str
    position: int
    sequence: str
    mismatches: int
    strand: str = "+"
    query_sequence: str = ""
    bulge_type: str = "X"
    bulge_size: int = 0
    query_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the canonical dictionary format used by agent tools."""
        return {
            "chromosome": self.chromosome,
            "position": self.position,
            "sequence": self.sequence,
            "mismatches": self.mismatches,
            "strand": self.strand,
            "query_sequence": self.query_sequence,
            "bulge_type": self.bulge_type,
            "bulge_size": self.bulge_size,
            "query_id": self.query_id,
        }


@dataclass
class CasOffinderRunResult:
    """Full result envelope returned by CasOffinderRunner.run()."""

    off_targets: list[dict[str, Any]] = field(default_factory=list)
    guide_sequence: str = ""
    genome_path: str = ""
    mismatch_threshold: int = 3
    hit_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "off_targets": self.off_targets,
            "hit_count": self.hit_count,
            "guide_sequence": self.guide_sequence,
            "genome_path": self.genome_path,
            "mismatch_threshold": self.mismatch_threshold,
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class CasOffinderRunner:
    """
    Production wrapper around the Cas-OFFinder command-line binary.

    Responsibilities (SRP):
      - Validate inputs
      - Write the Cas-OFFinder plain-text input file
      - Execute the binary via subprocess
      - Parse tab-separated output into structured records
      - Clean up all temporary files

    Dependencies (DIP): binary path and device mode are injected, not hard-coded.
    """

    # SpCas9 default: 20-bp protospacer + NRG PAM (23 positions total).
    DEFAULT_PAM_PATTERN: Final[str] = "NNNNNNNNNNNNNNNNNNNNNNRG"

    def __init__(
        self,
        binary_path: str | Path = "cas-offinder",
        device: str = "C",
        pam_pattern: str = DEFAULT_PAM_PATTERN,
        bulge_dna: int = 0,
        bulge_rna: int = 0,
        query_id: str = "guide1",
        timeout_seconds: int = 600,
    ) -> None:
        """
        Args:
            binary_path: Path to cas-offinder executable (or name on PATH).
            device: Hardware backend — G (GPU), C (CPU), or A (accelerator).
            pam_pattern: PAM/search pattern line for the input file.
            bulge_dna: DNA bulge size appended to the pattern line.
            bulge_rna: RNA bulge size appended to the pattern line.
            query_id: Optional identifier included in output for this guide.
            timeout_seconds: Max seconds to wait for subprocess completion.
        """
        self._binary_path = Path(binary_path)
        self._device = device.upper()
        self._pam_pattern = pam_pattern.upper()
        self._bulge_dna = bulge_dna
        self._bulge_rna = bulge_rna
        self._query_id = query_id
        self._timeout_seconds = timeout_seconds

        if self._device not in {"G", "C", "A"}:
            raise CasOffinderConfigError(
                f"Invalid device '{device}'. Cas-OFFinder expects G, C, or A."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        guide_sequence: str,
        genome_fasta_path: str | Path,
        mismatch_threshold: int = 3,
    ) -> CasOffinderRunResult:
        """
        Execute a full Cas-OFFinder search for a single guide sequence.

        Args:
            guide_sequence: Guide RNA / protospacer sequence (typically 20 bp).
            genome_fasta_path: Path to a FASTA file or directory of FASTA/2bit
                files (Cas-OFFinder requires a directory on line 1 of the input).
            mismatch_threshold: Maximum allowed mismatches (inclusive).

        Returns:
            CasOffinderRunResult with parsed off-target hits.

        Raises:
            CasOffinderConfigError: Invalid inputs.
            CasOffinderBinaryNotFoundError: Binary not found.
            CasOffinderExecutionError: Subprocess failure.
            CasOffinderParseError: Unparseable output.
        """
        guide = self._normalize_guide(guide_sequence)
        genome_dir = self._resolve_genome_directory(Path(genome_fasta_path))
        query_sequence = self._build_query_sequence(guide)

        if mismatch_threshold < 0:
            raise CasOffinderConfigError("mismatch_threshold must be >= 0.")

        self._ensure_binary_available()

        input_path: Path | None = None
        output_path: Path | None = None

        try:
            input_path, output_path = self._create_temp_paths()
            self._write_input_file(
                input_path=input_path,
                genome_dir=genome_dir,
                query_sequence=query_sequence,
                mismatch_threshold=mismatch_threshold,
            )
            self._execute(input_path=input_path, output_path=output_path)
            hits = self._parse_output(output_path)

            off_targets = [hit.to_dict() for hit in hits]
            return CasOffinderRunResult(
                off_targets=off_targets,
                guide_sequence=guide,
                genome_path=str(genome_dir),
                mismatch_threshold=mismatch_threshold,
                hit_count=len(off_targets),
            )
        finally:
            self._cleanup(input_path, output_path)

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_guide(guide_sequence: str) -> str:
        """Uppercase, strip whitespace, and validate nucleotide alphabet."""
        guide = guide_sequence.upper().strip()
        if not guide:
            raise CasOffinderConfigError("guide_sequence must not be empty.")
        if not _DNA_PATTERN.match(guide):
            raise CasOffinderConfigError(
                f"guide_sequence contains invalid characters: {guide!r}"
            )
        return guide

    @staticmethod
    def _resolve_genome_directory(genome_path: Path) -> Path:
        """
        Cas-OFFinder expects a *directory* on the first line of its input file.

        Accept either a directory directly or a FASTA file — in the latter case
        we use the parent directory.
        """
        resolved = genome_path.expanduser().resolve()

        if resolved.is_dir():
            return resolved

        if resolved.is_file():
            return resolved.parent

        raise CasOffinderConfigError(
            f"Genome path does not exist: {genome_path}"
        )

    def _build_query_sequence(self, guide: str) -> str:
        """
        Pad the guide with N's so its length matches the PAM pattern length.

        Cas-OFFinder requires query sequences to be the same length as the
        pattern line (protospacer + PAM positions).
        """
        pattern_len = len(self._pam_pattern)
        if len(guide) > pattern_len:
            raise CasOffinderConfigError(
                f"Guide length ({len(guide)} bp) exceeds PAM pattern length "
                f"({pattern_len} bp)."
            )
        pam_padding = "N" * (pattern_len - len(guide))
        return guide + pam_padding

    def _write_input_file(
        self,
        input_path: Path,
        genome_dir: Path,
        query_sequence: str,
        mismatch_threshold: int,
    ) -> None:
        """
        Write the Cas-OFFinder plain-text input file.

        Format:
            Line 1: genome directory path
            Line 2: PAM pattern [DNA_bulge] [RNA_bulge]
            Line 3+: <query_sequence> <max_mismatches> [optional_id]
        """
        pattern_line = self._pam_pattern
        if self._bulge_dna or self._bulge_rna:
            pattern_line = f"{pattern_line} {self._bulge_dna} {self._bulge_rna}"

        lines = [
            str(genome_dir),
            pattern_line,
            f"{query_sequence} {mismatch_threshold} {self._query_id}",
        ]

        try:
            input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logger.debug("Cas-OFFinder input written to %s", input_path)
        except OSError as exc:
            raise CasOffinderExecutionError(
                f"Failed to write Cas-OFFinder input file: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Subprocess execution
    # ------------------------------------------------------------------

    def _ensure_binary_available(self) -> None:
        """Verify the binary exists before attempting execution."""
        if self._binary_path.is_file():
            return
        if shutil.which(str(self._binary_path)):
            return
        raise CasOffinderBinaryNotFoundError(
            f"cas-offinder binary not found at '{self._binary_path}' "
            "and not present on system PATH. "
            "Install Cas-OFFinder or set CAS_OFFINDER_BINARY."
        )

    def _resolve_binary_command(self) -> str:
        """Return the executable path to pass to subprocess."""
        if self._binary_path.is_file():
            return str(self._binary_path)
        found = shutil.which(str(self._binary_path))
        if found:
            return found
        return str(self._binary_path)

    def _execute(self, input_path: Path, output_path: Path) -> None:
        """Invoke cas-offinder via subprocess.run with strict error checking."""
        command = [
            self._resolve_binary_command(),
            str(input_path),
            self._device,
            str(output_path),
        ]

        logger.info("Executing Cas-OFFinder: %s", " ".join(command))

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CasOffinderExecutionError(
                f"Cas-OFFinder timed out after {self._timeout_seconds}s."
            ) from exc
        except OSError as exc:
            raise CasOffinderExecutionError(
                f"Failed to launch Cas-OFFinder subprocess: {exc}"
            ) from exc

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or f"exit code {completed.returncode}"
            raise CasOffinderExecutionError(
                f"Cas-OFFinder exited with code {completed.returncode}: {detail}"
            )

        if not output_path.exists():
            raise CasOffinderExecutionError(
                "Cas-OFFinder completed but no output file was produced."
            )

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_output(self, output_path: Path) -> list[OffTargetHit]:
        """Parse the tab-separated Cas-OFFinder output into OffTargetHit records."""
        try:
            raw_text = output_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise CasOffinderParseError(
                f"Unable to read Cas-OFFinder output: {exc}"
            ) from exc

        if not raw_text:
            logger.info("Cas-OFFinder returned no off-target hits.")
            return []

        hits: list[OffTargetHit] = []
        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split("\t")
            try:
                hits.append(self._fields_to_hit(fields))
            except (CasOffinderParseError, ValueError) as exc:
                raise CasOffinderParseError(
                    f"Failed to parse output line {line_number}: {exc}"
                ) from exc

        return hits

    def _fields_to_hit(self, fields: list[str]) -> OffTargetHit:
        """Map a TSV row to an OffTargetHit, supporting modern and legacy formats."""
        column_count = len(fields)

        if column_count >= len(_MODERN_OUTPUT_COLUMNS):
            return OffTargetHit(
                query_id=fields[0],
                bulge_type=fields[1],
                query_sequence=fields[2],
                sequence=fields[3],
                chromosome=fields[4],
                position=int(fields[5]),
                strand=fields[6],
                mismatches=int(fields[7]),
                bulge_size=int(fields[8]) if fields[8] else 0,
            )

        if column_count >= len(_LEGACY_OUTPUT_COLUMNS):
            return OffTargetHit(
                query_sequence=fields[0],
                chromosome=fields[1],
                position=int(fields[2]),
                sequence=fields[3],
                strand=fields[4],
                mismatches=int(fields[5]),
            )

        raise CasOffinderParseError(
            f"Unexpected column count ({column_count}); expected "
            f"{len(_LEGACY_OUTPUT_COLUMNS)} (legacy) or "
            f"{len(_MODERN_OUTPUT_COLUMNS)} (modern)."
        )

    # ------------------------------------------------------------------
    # Temp file lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _create_temp_paths() -> tuple[Path, Path]:
        """Create uniquely named temporary input/output paths."""
        tmp_dir = Path(tempfile.gettempdir())
        input_fd, input_name = tempfile.mkstemp(
            prefix="cas_offinder_in_", suffix=".txt", dir=tmp_dir
        )
        output_fd, output_name = tempfile.mkstemp(
            prefix="cas_offinder_out_", suffix=".txt", dir=tmp_dir
        )

        # mkstemp opens the files; close immediately — Cas-OFFinder needs paths only.
        import os

        os.close(input_fd)
        os.close(output_fd)

        return Path(input_name), Path(output_name)

    @staticmethod
    def _cleanup(*paths: Path | None) -> None:
        """Best-effort removal of temporary files created during a run."""
        for path in paths:
            if path is None:
                continue
            try:
                path.unlink(missing_ok=True)
                logger.debug("Cleaned up temp file: %s", path)
            except OSError as exc:
                logger.warning("Failed to remove temp file %s: %s", path, exc)
