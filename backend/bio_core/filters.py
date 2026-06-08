from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict


NUC = ("A", "C", "G", "T")


@dataclass
class GuideFilterReport:
    guide_seq: str
    gc_content: float
    shannon_entropy: float
    has_homopolymer: bool
    has_polyT_u6: bool
    self_comp_max: int
    viable: bool
    reasons: str
    score: float  # Biophysical score in [0, 1]

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["gc_content"] = round(self.gc_content, 3)
        d["shannon_entropy"] = round(self.shannon_entropy, 3)
        d["score"] = round(self.score, 3)
        return d


def gc_fraction(seq: str) -> float:
    """Calculate the fraction of G and C nucleotides in the sequence."""
    if not seq:
        return 0.0
    cleaned = seq.upper()
    return (cleaned.count("G") + cleaned.count("C")) / len(cleaned)


def calculate_shannon_entropy(seq: str) -> float:
    """
    Calculate Shannon entropy as a measure of sequence complexity.
    Higher values = better balance of A/C/G/T.
    """
    from collections import Counter

    if not seq:
        return 0.0

    counts = Counter(seq)
    probs = [c / len(seq) for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def longest_self_complementarity(seq: str) -> int:
    """
    Measure the maximum length of a complementary reverse match (reverse-complement)
    within the guide sequence to assess hairpin/self-dimer risks.
    """
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    rc = "".join(comp.get(b, "N") for b in reversed(seq))

    n = len(seq)
    max_len = 0

    # Sliding window search for the longest substring of seq that appears in rc
    for w in range(4, n + 1):  # Focus on window lengths >= 4
        for i in range(0, n - w + 1):
            sub = seq[i : i + w]
            if sub in rc:
                max_len = max(max_len, w)
    return max_len


def evaluate_guide(
    guide_seq: str,
    promoter: str = "U6",
    min_gc: float = 0.35,
    max_gc: float = 0.65,
    min_entropy: float = 1.5,
    max_self_comp: int = 6,
) -> GuideFilterReport:
    """
    Perform biological viability checks on a 20bp guide sequence.

    Checks:
    - GC content in [min_gc, max_gc] (default 35-65%).
    - Shannon entropy complexity >= min_entropy (default 1.5).
    - No homopolymer runs >= 5 nt (e.g. AAAAA).
    - No Poly-T termination motif (TTTT or TTTTT) under U6 promoter.
    - Self-complementarity <= max_self_comp (default 6 nt).
    """
    seq = guide_seq.upper().strip()

    # GC and Shannon entropy
    gc = gc_fraction(seq)
    entropy = calculate_shannon_entropy(seq)

    # Homopolymer runs
    has_homopolymer = any(n * 5 in seq for n in NUC)

    # Poly-T termination check
    has_polyT_u6 = False
    if promoter.upper() == "U6":
        if "TTTTT" in seq or "TTTT" in seq:
            has_polyT_u6 = True

    # Self-complementarity
    self_comp = longest_self_complementarity(seq)

    reasons = []
    score = 1.0

    # Check constraints and deduct score points
    if not (min_gc <= gc <= max_gc):
        reasons.append("GC out of preferred range")
        score -= 0.25

    if entropy < min_entropy:
        reasons.append("Low Shannon entropy (low complexity)")
        score -= 0.25

    if has_homopolymer:
        reasons.append("Homopolymer run >= 5 nt")
        score -= 0.25

    if has_polyT_u6:
        reasons.append("Poly-T motif (U6 termination risk)")
        score -= 0.2

    if self_comp > max_self_comp:
        reasons.append(f"High self-complementarity (max match {self_comp} nt)")
        score -= 0.25

    viable = (len(reasons) == 0) and score > 0.0
    if viable:
        reasons.append("OK")

    # Clip score to [0.0, 1.0]
    score = max(min(score, 1.0), 0.0)

    return GuideFilterReport(
        guide_seq=seq,
        gc_content=gc,
        shannon_entropy=entropy,
        has_homopolymer=has_homopolymer,
        has_polyT_u6=has_polyT_u6,
        self_comp_max=self_comp,
        viable=viable,
        reasons="; ".join(reasons),
        score=score,
    )
