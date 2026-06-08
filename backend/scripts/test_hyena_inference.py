#!/usr/bin/env python3
"""
Verification script for HyenaZeroShotEvaluator singleton and thread-safety.

Tests:
1. Singleton identity check.
2. Model and tokenizer loading.
3. Zero-shot efficiency score bounds [0.0, 1.0].
4. Thread-safety under concurrent inference requests.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from bio_core.hyena_inference import HyenaZeroShotEvaluator, HyenaDNAInferenceError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_hyena")


def test_singleton_identity() -> None:
    logger.info("Running Singleton Identity Test...")
    eval1 = HyenaZeroShotEvaluator("LongSafari/hyenadna-tiny-1k-seqlen-hf")
    eval2 = HyenaZeroShotEvaluator("LongSafari/hyenadna-tiny-1k-seqlen-hf")
    
    assert eval1 is eval2, "FAIL: HyenaZeroShotEvaluator is not a singleton!"
    logger.info("PASS: Singleton identity verified (eval1 is eval2).")


def test_model_loading_and_bounds() -> None:
    logger.info("Running Model Loading and Bounds Test...")
    evaluator = HyenaZeroShotEvaluator("LongSafari/hyenadna-tiny-1k-seqlen-hf")
    
    start_time = time.time()
    evaluator.load_model()
    logger.info("Model loaded in %.2f seconds.", time.time() - start_time)
    
    assert evaluator.model is not None, "FAIL: Model was not loaded."
    assert evaluator.tokenizer is not None, "FAIL: Tokenizer was not loaded."
    
    # Test prediction on a short DNA sequence
    test_seq = "ATCGATCGATCGATCGATCG"
    score = evaluator.predict_efficiency(test_seq)
    logger.info("Sequence: %s -> Predicted Score: %f", test_seq, score)
    
    assert isinstance(score, float), "FAIL: Score is not a float."
    assert 0.0 <= score <= 1.0, f"FAIL: Score {score} is out of bounds [0.0, 1.0]."
    logger.info("PASS: Model loading and score bounds verified.")


def test_thread_safety() -> None:
    logger.info("Running Thread Safety Test (Concurrent Inference)...")
    evaluator = HyenaZeroShotEvaluator("LongSafari/hyenadna-tiny-1k-seqlen-hf")
    
    sequences = [
        "ATCGATCGATCGATCGATCG",
        "GCTAGCTAGCTAGCTAGCTA",
        "AAAAATTTTTGGGGGCCCCC",
        "CCTTGGAACCTTGGAACCTT",
        "NNNNNATCGATCGNNNNNTA"
    ]
    
    results: list[float | Exception] = [0.0] * len(sequences)
    
    def worker(index: int, seq: str) -> None:
        logger.info("Thread starting inference for: %s", seq)
        try:
            score = evaluator.predict_efficiency(seq)
            results[index] = score
            logger.info("Thread finished: %s -> %f", seq, score)
        except Exception as e:
            results[index] = e
            logger.error("Thread failed: %s -> %s", seq, e)

    threads = []
    for i, seq in enumerate(sequences):
        t = threading.Thread(target=worker, args=(i, seq), name=f"InferenceWorker-{i}")
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    # Check results
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            raise AssertionError(f"FAIL: Thread {i} raised an exception: {res}")
        assert 0.0 <= res <= 1.0, f"FAIL: Thread {i} returned out-of-bounds score: {res}"
        
    logger.info("PASS: Thread safety verified. All %d concurrent threads finished successfully.", len(sequences))


def main() -> int:
    try:
        test_singleton_identity()
        print("-" * 50)
        test_model_loading_and_bounds()
        print("-" * 50)
        test_thread_safety()
        print("-" * 50)
        logger.info("ALL TESTS PASSED SUCCESSFULLY!")
        return 0
    except AssertionError as e:
        logger.error("TEST ASSERTION FAILED: %s", e)
        return 1
    except Exception as e:
        logger.error("UNEXPECTED ERROR DURING TESTING: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
