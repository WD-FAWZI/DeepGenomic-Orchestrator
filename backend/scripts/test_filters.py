import sys
from pathlib import Path

# Add backend directory to sys.path so we can import from bio_core and agent
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from bio_core.filters import evaluate_guide, gc_fraction, calculate_shannon_entropy, longest_self_complementarity

def test_gc_fraction():
    print("Testing gc_fraction...")
    assert gc_fraction("ATCGATCG") == 0.5
    assert gc_fraction("AAAAAA") == 0.0
    assert gc_fraction("GGGGGG") == 1.0
    assert gc_fraction("atcg") == 0.5
    print("PASS: gc_fraction calculations verified.")

def test_shannon_entropy():
    print("Testing calculate_shannon_entropy...")
    assert calculate_shannon_entropy("AAAAAA") == 0.0
    # Equal distribution of A, C, G, T should have max entropy
    ent_4 = calculate_shannon_entropy("ACGT")
    assert ent_4 == 2.0
    print("PASS: Shannon entropy calculations verified.")

def test_self_complementarity():
    print("Testing longest_self_complementarity...")
    # Since window search focuses on w >= 4, a non-self-comp seq of length 4 has match length 0
    assert longest_self_complementarity("ATCG") == 0
    assert longest_self_complementarity("ATCGAT") == 6
    assert longest_self_complementarity("AAAA") == 0
    print("PASS: Self-complementarity calculations verified.")

def test_evaluate_guide():
    print("Testing evaluate_guide...")
    # 1. Viable guide
    viable_seq = "ACCGTGTCAGTCAGTCAACA"  # GC = 50%, no homopolymers, no poly-T, low self-comp
    report = evaluate_guide(viable_seq)
    assert report.viable is True
    assert report.score == 1.0
    assert report.reasons == "OK"

    # 2. Homopolymer failure
    homopolymer_seq = "AAAAATCGATCGATCGATCG"
    report = evaluate_guide(homopolymer_seq)
    assert report.viable is False
    assert "Homopolymer run >= 5 nt" in report.reasons

    # 3. Poly-T failure (U6)
    polyT_seq = "ATCGATCGTTTTATCGATCG"
    report = evaluate_guide(polyT_seq, promoter="U6")
    assert report.viable is False
    assert "Poly-T motif (U6 termination risk)" in report.reasons

    # 4. Poly-T ignored under non-U6 promoter
    report = evaluate_guide(polyT_seq, promoter="other")
    assert report.viable is True  # Should pass since promoter is not U6

    # 5. Out of bounds GC content
    high_gc_seq = "GGGGGGGGGGGGGGGGGGGG"
    report = evaluate_guide(high_gc_seq)
    assert report.viable is False
    assert "GC out of preferred range" in report.reasons

    print("PASS: evaluate_guide constraints and scoring verified.")

if __name__ == "__main__":
    print("Running biophysical filter tests...")
    try:
        test_gc_fraction()
        test_shannon_entropy()
        test_self_complementarity()
        test_evaluate_guide()
        print("\nALL TESTS PASSED SUCCESSFULLY!")
    except AssertionError as e:
        import traceback
        print("\nTEST FAILED:")
        traceback.print_exc()
        sys.exit(1)
