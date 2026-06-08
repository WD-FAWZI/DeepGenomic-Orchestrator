import asyncio
import os
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Configure environment for testing
os.environ["CAS_OFFINDER_GENOME_PATH"] = "genome/mock_genome.fa"
os.environ["CAS_OFFINDER_DEVICE"] = "C"  # Use CPU to avoid OpenCL GPU compilation latency in tests

from agent.graph import run_evaluation
from agent.tools import warmup_hyenadna_model

async def run_one(task_id: int, sequence: str) -> dict:
    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Task {task_id} - Starting evaluation for: {sequence[:20]}...")
    
    # Run evaluation (asynchronously)
    result = await run_evaluation(sequence)
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[{time.strftime('%H:%M:%S')}] Task {task_id} - Finished in {elapsed:.3f}s. "
          f"Viable: {result.biological_filter_result['viable']}, "
          f"Cas-OFFinder run: {result.cas_offinder_result is not None}, "
          f"HyenaDNA Score: {result.hyenadna_score}")
    
    return {
        "task_id": task_id,
        "elapsed": elapsed,
        "result": result
    }

async def main():
    print("==================================================")
    print("Starting E2E Async Concurrent Inference Test...")
    print("==================================================")
    
    # Warm up model to load weights into memory so it doesn't skew concurrent timings
    print("Warming up HyenaDNA model...")
    warmup_hyenadna_model()
    print("Warmup complete.")
    
    # We will send 3 concurrent requests:
    # - Task 1: Viable (GC 50%, no homopolymers, no poly-T, low self-comp) -> full pipeline
    # - Task 2: Viable (GC 50%, no homopolymers, no poly-T, low self-comp) -> full pipeline (will serialize inference)
    # - Task 3: Non-viable (Homopolymer run) -> should short-circuit immediately!
    
    seq_viable1 = "ACCGTGTCAGTCAGTCAACA"
    seq_viable2 = "ACCGTGTCAGTCAGTCAACA"
    seq_non_viable = "ATCGATCGATCGAAAAAATCG"
    
    overall_start = time.time()
    
    print("\nLaunching 3 concurrent evaluations via asyncio.gather()...\n")
    results = await asyncio.gather(
        run_one(1, seq_viable1),
        run_one(2, seq_viable2),
        run_one(3, seq_non_viable)
    )
    
    overall_end = time.time()
    overall_elapsed = overall_end - overall_start
    print(f"\nAll tasks finished. Total elapsed time: {overall_elapsed:.3f}s\n")
    
    # Assertions
    task1 = next(r for r in results if r["task_id"] == 1)
    task2 = next(r for r in results if r["task_id"] == 2)
    task3 = next(r for r in results if r["task_id"] == 3)
    
    # Task 3 should short-circuit and run much faster than the viable ones
    assert task3["result"].biological_filter_result["viable"] is False, "Task 3 should be non-viable"
    assert task3["result"].cas_offinder_result is None, "Task 3 should bypass Cas-OFFinder"
    assert task3["result"].hyenadna_score is None, "Task 3 should bypass HyenaDNA"
    
    # Tasks 1 & 2 should complete fully
    assert task1["result"].biological_filter_result["viable"] is True, "Task 1 should be viable"
    assert task1["result"].cas_offinder_result is not None, "Task 1 should run Cas-OFFinder"
    assert task1["result"].hyenadna_score is not None, "Task 1 should get HyenaDNA score"
    
    assert task2["result"].biological_filter_result["viable"] is True, "Task 2 should be viable"
    assert task2["result"].cas_offinder_result is not None, "Task 2 should run Cas-OFFinder"
    assert task2["result"].hyenadna_score is not None, "Task 2 should get HyenaDNA score"
    
    print("==================================================")
    print("E2E CONCURRENT GRAPH FLOWS VERIFIED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(main())
