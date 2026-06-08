"""
Environment loading and Cas-OFFinder setup validation.

Paths in .env are resolved relative to the backend/ root directory so the
application behaves consistently regardless of the process working directory.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

# backend/ directory — anchor for all relative path resolution.
BACKEND_ROOT: Path = Path(__file__).resolve().parent.parent

ENV_BINARY_KEY = "CAS_OFFINDER_BINARY"
ENV_GENOME_KEY = "CAS_OFFINDER_GENOME_PATH"
ENV_HYENA_MODEL_KEY = "HYENA_MODEL_PATH"
DEFAULT_BINARY = "./bin/cas-offinder.exe"
DEFAULT_GENOME = "./genome/hg38.fa"


class SetupStatus(TypedDict):
    """Result of a Cas-OFFinder environment validation check."""

    ready: bool
    binary_path: str
    genome_path: str
    binary_exists: bool
    genome_exists: bool
    binary_on_path: bool
    missing: list[str]
    hyena_model_path: str
    hyena_model_exists: bool


def load_environment() -> None:
    """
    Load backend/.env into os.environ (if present).

    Uses python-dotenv when available; silently continues if the file or
    package is missing so the app never crashes during setup.
    """
    env_file = BACKEND_ROOT / ".env"
    if not env_file.is_file():
        logger.debug("No .env file found at %s — using system environment.", env_file)
        return

    try:
        from dotenv import load_dotenv

        loaded = load_dotenv(env_file, override=False)
        if loaded:
            logger.info("Loaded environment from %s", env_file)
        else:
            logger.debug("dotenv found no new variables in %s", env_file)
    except ImportError:
        logger.warning(
            "python-dotenv is not installed — copy backend/.env.example to "
            "backend/.env and export variables manually, or run: pip install python-dotenv"
        )


def resolve_env_path(
    raw_path: str | None,
    *,
    default: str | None = None,
) -> Path:
    """
    Resolve an environment path to an absolute Path.

    Relative paths are resolved against BACKEND_ROOT, not cwd.
    """
    value = (raw_path or default or "").strip()
    if not value:
        return Path()

    path = Path(value)
    if not path.is_absolute():
        path = (BACKEND_ROOT / path).resolve()
    return path


def _path_exists(path: Path) -> bool:
    """Check file or directory existence defensively."""
    try:
        return path.exists()
    except OSError as exc:
        logger.warning("Unable to stat path %s: %s", path, exc)
        return False


def check_cas_offinder_setup() -> SetupStatus:
    """
    Validate Cas-OFFinder binary and genome paths from the environment.

    Returns a structured status dict — never raises.
    """
    binary_raw = os.getenv(ENV_BINARY_KEY, DEFAULT_BINARY)
    genome_raw = os.getenv(ENV_GENOME_KEY, DEFAULT_GENOME)
    hyena_model_raw = os.getenv(ENV_HYENA_MODEL_KEY, "")

    binary_path = resolve_env_path(binary_raw, default=DEFAULT_BINARY)
    genome_path = resolve_env_path(genome_raw, default=DEFAULT_GENOME)
    hyena_model_path = resolve_env_path(hyena_model_raw) if hyena_model_raw else Path()

    binary_exists = _path_exists(binary_path)
    binary_on_path = bool(shutil.which(str(binary_raw)) or shutil.which(str(binary_path)))
    genome_exists = _path_exists(genome_path)
    hyena_model_exists = bool(hyena_model_path) and _path_exists(hyena_model_path)

    binary_ready = binary_exists or binary_on_path
    missing: list[str] = []

    if not binary_ready:
        missing.append(f"binary ({binary_path})")
    if not genome_exists:
        missing.append(f"genome ({genome_path})")
    if not hyena_model_exists:
        missing.append("HYENA_MODEL_PATH (set to local model weights/config)")

    status: SetupStatus = {
        "ready": binary_ready and genome_exists and hyena_model_exists,
        "binary_path": str(binary_path),
        "genome_path": str(genome_path),
        "binary_exists": binary_exists,
        "genome_exists": genome_exists,
        "binary_on_path": binary_on_path,
        "missing": missing,
        "hyena_model_path": str(hyena_model_path),
        "hyena_model_exists": hyena_model_exists,
    }

    return status


def _console_print(message: str) -> None:
    """
    Print to the terminal with a safe fallback for Windows code pages.

    Avoids crashing startup when the console cannot render emoji / ANSI.
    """
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="replace").decode("ascii"))


def print_setup_warning(status: SetupStatus | None = None) -> None:
    """
    Emit a stylized console warning when Cas-OFFinder is not fully configured.

    Uses print() (not logging) so the message is always visible in the
    uvicorn terminal during local development.
    """
    status = status or check_cas_offinder_setup()

    if status["ready"]:
        _console_print(
            "\033[92m✔ [SETUP OK]\033[0m Cas-OFFinder + HyenaDNA model assets detected."
        )
        _console_print(f"   Binary : {status['binary_path']}")
        _console_print(f"   Genome : {status['genome_path']}")
        _console_print(f"   Hyena  : {status['hyena_model_path']}")
        return

    _console_print(
        "\033[93m⚠️ [SETUP REQUIRED]\033[0m Cas-OFFinder binary, Genome FASTA, or HyenaDNA model "
        "missing. Pipeline will fall back to mock data or fail gracefully."
    )

    for item in status["missing"]:
        _console_print(f"   ✗ Missing {item}")

    _console_print("")
    _console_print("   Quick-start:")
    _console_print("   1. Copy backend/.env.example -> backend/.env")
    _console_print("   2. Place cas-offinder.exe in backend/bin/")
    _console_print("   3. Run: python scripts/generate_mock_genome.py  (for test FASTA)")
    _console_print("   4. Set CAS_OFFINDER_GENOME_PATH=./genome/mock_genome.fa in .env")
    _console_print("   5. Set HYENA_MODEL_PATH=/absolute/or/relative/path/to/hyena/model")
    _console_print("")
