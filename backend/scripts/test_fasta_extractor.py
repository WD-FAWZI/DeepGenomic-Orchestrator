import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from bio_core.fasta_extractor import FastaExtractor, reverse_complement

def test_fasta_extractor():
    genome_file = backend_dir / "genome" / "mock_genome.fa"
    print(f"Testing FastaExtractor with: {genome_file}")
    
    extractor = FastaExtractor(genome_file)
    
    # Check that mock_chr1 is indexed
    print(f"Index keys: {list(extractor.index.keys())}")
    assert "mock_chr1" in extractor.index, "mock_chr1 not found in index"
    
    info = extractor.index["mock_chr1"]
    print(f"mock_chr1 metadata: {info}")
    assert info["line_len"] == 80, f"Expected line_len 80, got {info['line_len']}"
    
    # Test extraction at position 0
    seq_0_20 = extractor.extract_sequence("mock_chr1", 0, 20)
    print(f"Extracted [0:20]: {seq_0_20}")
    assert seq_0_20 == "ATCGATCGATCGATCGATCG", f"Expected ATCGATCGATCGATCGATCG, got {seq_0_20}"
    
    # Test extraction spanning a line break (around pos 80)
    # Positions 75 to 85. Line length is 80, so index 80 starts the second line.
    seq_line_break = extractor.extract_sequence("mock_chr1", 75, 10)
    print(f"Extracted [75:85] (cross-line): {seq_line_break}")
    assert seq_line_break == "GATCGATCGA", f"Expected GATCGATCGA, got {seq_line_break}"
    
    # Test clamping/padding for negative coordinates
    seq_neg = extractor.extract_sequence("mock_chr1", -5, 10)
    print(f"Extracted [-5:5] (negative offset): {seq_neg}")
    assert seq_neg == "NNNNNATCGA", f"Expected NNNNNATCGA, got {seq_neg}"
    
    # Test clamping/padding for coordinates past the end
    total_len = info["total_len"]
    seq_end = extractor.extract_sequence("mock_chr1", total_len - 5, 10)
    print(f"Extracted [end-5:end+5]: {seq_end}")
    assert len(seq_end) == 10
    assert seq_end.endswith("NNNNN")
    
    # Test reverse complement
    rc = reverse_complement("ATCGN")
    print(f"Reverse complement of ATCGN: {rc}")
    assert rc == "NCGAT"
    
    print("ALL FASTA EXTRACTOR TESTS PASSED!")

if __name__ == "__main__":
    test_fasta_extractor()
