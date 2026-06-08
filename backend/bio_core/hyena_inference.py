"""
HyenaDNA inference wrapper for zero-shot sequence evaluation.

Provides the thread-safe HyenaZeroShotEvaluator singleton to run local-first
inference using Hugging Face's pretrained HyenaDNA checkpoints.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class HyenaDNAInferenceError(Exception):
    """Raised when the HyenaDNA inference pipeline encounters a configuration or execution error."""
    pass


class HyenaZeroShotEvaluator:
    """
    Thread-safe Singleton class for HyenaDNA zero-shot sequence efficiency evaluation.

    Encapsulates model initialization, checkpoint loading, tokenizer management,
    and concurrent model forward passes.
    """

    _instance: HyenaZeroShotEvaluator | None = None
    _singleton_lock: threading.Lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> HyenaZeroShotEvaluator:
        """Double-checked locking pattern for thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: str | Path | None = None, device: str | None = None) -> None:
        """
        Initialize the evaluator configurations.

        The actual model loading is deferred to load_model() or the first prediction.
        """
        if getattr(self, "_initialized", False):
            return

        # Use environment variables if not passed explicitly
        raw_path = str(model_path) if model_path else os.getenv("HYENA_MODEL_PATH", "").strip()
        self.model_path = raw_path or "LongSafari/hyenadna-tiny-1k-seqlen-hf"

        # Determine compute backend (CUDA / CPU / MPS)
        if device:
            self.device = torch.device(device)
        else:
            env_device = os.getenv("HYENA_DEVICE", "").strip()
            if env_device:
                self.device = torch.device(env_device)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.tokenizer = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._initialized = True
        logger.info(
            "HyenaZeroShotEvaluator initialized (deferred loading). Target device: %s, Model path: %s",
            self.device,
            self.model_path,
        )

    def load_model(self) -> None:
        """
        Thread-safe method to load the model and tokenizer into memory/GPU.

        Uses double-checked locking to prevent concurrent load attempts from multiple threads.
        """
        if self.model is not None:
            return  # Model already loaded

        with self._load_lock:
            if self.model is not None:
                return  # Double check inside lock

            logger.info("Loading HyenaDNA model and tokenizer from %s...", self.model_path)
            try:
                # Resolve path if it points to a local directory
                target_path = self.model_path
                if isinstance(target_path, str) and (target_path.startswith("./") or target_path.startswith("../")):
                    # Resolve relative to backend root if possible
                    from config.environment import resolve_env_path
                    resolved = resolve_env_path(target_path)
                    if resolved.exists():
                        target_path = str(resolved)

                # Load tokenizer (requires trust_remote_code=True for custom Hyena tokenization)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    target_path,
                    trust_remote_code=True,
                )

                # Load sequence classification model (requires trust_remote_code=True)
                # We use AutoModelForSequenceClassification so that a classification head is available
                # or can be initialized for zero-shot scoring.
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    target_path,
                    trust_remote_code=True,
                ).to(self.device)

                self.model.eval()
                logger.info("Successfully loaded HyenaDNA model on device: %s", self.device)

            except Exception as exc:
                # Reset to None to allow subsequent retry attempts if loading failed
                self.model = None
                self.tokenizer = None
                logger.error("Failed to load HyenaDNA model from %s: %s", self.model_path, exc)
                raise HyenaDNAInferenceError(f"Failed to load HyenaDNA model: {exc}") from exc

    def predict_efficiency(self, sequence: str) -> float:
        """
        Perform a thread-safe model forward pass to compute CRISPR efficiency score in [0, 1].

        Args:
            sequence: Raw DNA nucleotide sequence (A, T, C, G, N).

        Returns:
            A floating-point score between 0.0 and 1.0 representing predicted efficiency.
        """
        # Ensure the model is loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()

        assert self.tokenizer is not None
        assert self.model is not None

        clean_seq = sequence.upper().strip()
        if not clean_seq:
            raise HyenaDNAInferenceError("DNA sequence cannot be empty.")

        try:
            with self._inference_lock:
                # Tokenize input
                inputs = self.tokenizer(clean_seq, return_tensors="pt")
                input_ids = inputs["input_ids"].to(self.device)

                with torch.inference_mode():
                    outputs = self.model(input_ids)
                    
                    # Retrieve logits
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits
                    elif isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        # Fallback if the return type is a generic ModelOutput without logits attribute
                        logits = getattr(outputs, "last_hidden_state", outputs)
                        if hasattr(logits, "mean"):
                            logits = logits.mean(dim=1)  # average pooling

                    # Map logits to [0, 1] efficiency score
                    # Case 1: Multi-class logits (e.g. active vs inactive) -> Softmax
                    if logits.shape[-1] > 1:
                        probs = torch.softmax(logits, dim=-1)
                        # Use class 1 (usually representing active/efficient guide)
                        score = probs[0, 1].item()
                    # Case 2: Single-logit regression/binary classification -> Sigmoid
                    else:
                        score = torch.sigmoid(logits[0, 0]).item()

                    # Ensure score is strictly clamped to [0.0, 1.0]
                    return float(max(0.0, min(1.0, score)))

        except Exception as exc:
            logger.error("Error during HyenaDNA inference for sequence %s: %s", sequence, exc)
            raise HyenaDNAInferenceError(f"Inference execution failed: {exc}") from exc


# Alias class to maintain backward compatibility with existing agent configurations
HyenaDNAInference = HyenaZeroShotEvaluator
