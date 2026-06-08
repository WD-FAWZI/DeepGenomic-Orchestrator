import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Configure environment for testing
temp_fa_relative = "genome/mock_genome_test_2kb.fa"
temp_fa_absolute = backend_dir / temp_fa_relative
temp_index_absolute = backend_dir / "genome/mock_genome_test_2kb.fa.index.json"

os.environ["CAS_OFFINDER_GENOME_PATH"] = temp_fa_relative
os.environ["CAS_OFFINDER_DEVICE"] = "C"

from agent.graph import run_evaluation
import agent.graph
from agent.tools import warmup_hyenadna_model

def mock_run_cas_offinder(guide_seq: str, genome_fasta_path: str | None = None, mismatch_threshold: int = 3) -> dict:
    return {
        "off_targets": [
            {
                "chromosome": "mock_chr_test",
                "position": 1000,
                "sequence": guide_seq + "CGG",
                "mismatches": 0,
                "strand": "+",
                "query_sequence": guide_seq + "GGG",
                "bulge_type": "X",
                "bulge_size": 0,
                "query_id": "guide1"
            }
        ],
        "hit_count": 1,
        "guide_sequence": guide_seq,
        "genome_path": str(temp_fa_absolute),
        "mismatch_threshold": mismatch_threshold
    }

agent.graph.run_cas_offinder = mock_run_cas_offinder

def setup_temp_genome():
    original_fa = backend_dir / "genome" / "mock_genome.fa"
    print(f"Creating temporary test genome from {original_fa} -> {temp_fa_absolute}")
    shutil.copy(original_fa, temp_fa_absolute)
    
    # 1000 bp upstream 'A', 20 bp guide + 3 bp PAM, 1000 bp downstream 'T'
    guide = "ACCGTGTCAGTCAGTCAACA"
    pam = "CGG"
    site = guide + pam # 23 bp
    upstream = "A" * 1000
    downstream = "T" * 1000
    full_seq = upstream + site + downstream # 2023 bp
    
    with open(temp_fa_absolute, "a") as f:
        f.write("\n>mock_chr_test test contig for 2kb context\n")
        # Wrap sequence at 80 characters
        for i in range(0, len(full_seq), 80):
            f.write(full_seq[i:i+80] + "\n")
    print("Temporary genome setup complete with chromosome 'mock_chr_test'.")

def cleanup_temp_genome():
    print("Cleaning up temporary test genome files...")
    if temp_fa_absolute.exists():
        temp_fa_absolute.unlink()
    if temp_index_absolute.exists():
        temp_index_absolute.unlink()
    print("Cleanup complete.")

async def main():
    print("==================================================")
    print("Starting 2kb Flanking Genomic Context E2E Test...")
    print("==================================================")
    
    setup_temp_genome()
    
    try:
        print("Warming up HyenaDNA model...")
        warmup_hyenadna_model()
        print("Warmup complete.")
        
        # Guide sequence (viable)
        seq = "ACCGTGTCAGTCAGTCAACA"
        
        print(f"Running evaluation for guide: {seq} against temporary genome...")
        start_time = time.time()
        result = await run_evaluation(seq)
        end_time = time.time()
        
        print(f"Evaluation completed in {end_time - start_time:.3f}s.")
        print(f"Viable: {result.biological_filter_result['viable']}")
        print(f"Cas-OFFinder run: {result.cas_offinder_result is not None}")
        print(f"Cas-OFFinder result: {result.cas_offinder_result}")
        print(f"HyenaDNA Score: {result.hyenadna_score}")
        print(f"Metadata: {result.metadata}")
        
        # Verify assertions
        assert result.biological_filter_result['viable'] is True, "Guide should be viable"
        assert result.cas_offinder_result is not None, "Cas-OFFinder should run successfully"
        assert result.hyenadna_score is not None, "HyenaDNA score should be generated"
        
        # Ensure context was extracted
        assert "extracted_context" in result.metadata, "Metadata should contain extracted_context info"
        ctx = result.metadata["extracted_context"]
        print(f"Extracted context info: {ctx}")
        assert ctx["chromosome"] == "mock_chr_test", f"Expected mock_chr_test, got {ctx['chromosome']}"
        assert ctx["position"] == 1000, f"Expected position 1000, got {ctx['position']}"
        assert ctx["strand"] == "+", f"Expected strand +, got {ctx['strand']}"
        assert ctx["length"] == 2023, f"Expected length 2023, got {ctx['length']}"
        
        print("\n==================================================")
        print("2kb FLANKING GENOMIC CONTEXT E2E TEST PASSED!")
        print("==================================================")
        
    finally:
        cleanup_temp_genome()

if __name__ == "__main__":
    asyncio.run(main())
