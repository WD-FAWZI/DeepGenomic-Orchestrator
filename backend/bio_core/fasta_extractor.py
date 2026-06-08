"""
FASTA extraction utility for high-throughput zero-dependency coordinate lookup.

Provides O(1) sequence extraction from chromosome-specific directories or indexed 
single-file genomes (using standard .fai index or custom .index.json cache).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

class FastaExtractor:
    """
    Fast, zero-dependency FASTA sequence reader supporting directory-based
    chromosomes and single-file genomes with byte-offset indexing.
    """

    def __init__(self, genome_path: str | Path) -> None:
        self.genome_path = Path(genome_path)
        self.is_dir = self.genome_path.is_dir()
        self.index: dict[str, dict[str, int]] = {}
        self.single_file_index_loaded = False

        if not self.is_dir:
            self.index_path = self.genome_path.with_suffix(self.genome_path.suffix + ".index.json")
            self._load_or_build_index()

    def _load_or_build_index(self) -> None:
        """Load an index from an existing .fai file or a cached JSON index, otherwise build it."""
        # 1. Try to load from standard .fai index
        fai_path = self.genome_path.with_suffix(self.genome_path.suffix + ".fai")
        if not fai_path.exists():
            fai_path2 = self.genome_path.parent / (self.genome_path.name + ".fai")
            if fai_path2.exists():
                fai_path = fai_path2

        if fai_path.exists():
            try:
                index = {}
                with open(fai_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 5:
                            chrom = parts[0]
                            total_len = int(parts[1])
                            offset = int(parts[2])
                            line_len = int(parts[3])
                            line_bytes = int(parts[4])
                            index[chrom] = {
                                "offset": offset,
                                "line_len": line_len,
                                "line_bytes": line_bytes,
                                "total_len": total_len,
                            }
                self.index = index
                self.single_file_index_loaded = True
                return
            except Exception:
                pass

        # 2. Try to load from the index.json cache
        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    self.index = json.load(f)
                self.single_file_index_loaded = True
                return
            except Exception:
                pass

        # 3. Scan and build index if neither exist
        self._build_index()

    def _build_index(self) -> None:
        """Scan the FASTA file once to generate chromosomal byte-offsets and line configurations."""
        index = {}
        current_chrom = None
        current_offset = None
        line_len = None
        line_bytes = None
        total_len = 0

        with open(self.genome_path, "rb") as f:
            offset = 0
            while True:
                line = f.readline()
                if not line:
                    break
                line_len_with_nl = len(line)
                if line.startswith(b">"):
                    # Save previous chromosome metadata
                    if current_chrom is not None:
                        index[current_chrom] = {
                            "offset": current_offset,
                            "line_len": line_len,
                            "line_bytes": line_bytes,
                            "total_len": total_len,
                        }
                    # Parse chromosome header
                    header = line.decode("utf-8", errors="ignore")
                    current_chrom = header[1:].strip().split()[0]
                    current_offset = offset + line_len_with_nl
                    line_len = None
                    line_bytes = None
                    total_len = 0
                else:
                    if current_chrom is not None:
                        stripped = line.rstrip(b"\r\n")
                        curr_len = len(stripped)
                        if curr_len > 0:
                            if line_len is None:
                                line_len = curr_len
                                line_bytes = line_len_with_nl
                            total_len += curr_len
                offset += line_len_with_nl

            # Save the final chromosome metadata
            if current_chrom is not None:
                index[current_chrom] = {
                    "offset": current_offset,
                    "line_len": line_len,
                    "line_bytes": line_bytes,
                    "total_len": total_len,
                }

        self.index = index
        self.single_file_index_loaded = True
        try:
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(index, f)
        except Exception:
            pass

    def _find_chrom_file_in_dir(self, chromosome: str) -> Path | None:
        """Find a chromosome-specific file in the genome directory."""
        chrom_lower = chromosome.lower()
        for entry in self.genome_path.iterdir():
            if entry.is_file() and entry.suffix.lower() in (".fa", ".fasta", ".fna"):
                stem_lower = entry.stem.lower()
                if (
                    stem_lower == chrom_lower
                    or stem_lower == f"chr{chrom_lower}"
                    or stem_lower.replace("chr", "") == chrom_lower.replace("chr", "")
                ):
                    return entry
        return None

    def _get_chrom_info_from_file(self, filepath: Path) -> dict[str, Any] | None:
        """Read FASTA header and first line to extract layout parameters dynamically."""
        try:
            with open(filepath, "rb") as f:
                header = f.readline()
                if not header.startswith(b">"):
                    return None
                offset = len(header)
                first_seq_line = f.readline()
                if not first_seq_line:
                    return None
                stripped = first_seq_line.rstrip(b"\r\n")
                line_len = len(stripped)
                line_bytes = len(first_seq_line)
                
                # Use file size as upper bound on sequence characters
                file_size = filepath.stat().st_size
                total_len = file_size - offset
                return {
                    "offset": offset,
                    "line_len": line_len,
                    "line_bytes": line_bytes,
                    "total_len": total_len,
                    "filepath": filepath,
                }
        except Exception:
            return None

    def extract_sequence(self, chromosome: str, position: int, length: int) -> str:
        """
        Extract sequence from chromosome at 0-indexed position of specified length.

        Handles boundary clamping and left/right padding with 'N's if coordinate ranges
        fall outside chromosome boundaries.
        """
        if self.is_dir:
            chrom_file = self._find_chrom_file_in_dir(chromosome)
            if not chrom_file:
                # Try with/without 'chr' prefix
                alt_chrom = chromosome.replace("chr", "") if chromosome.lower().startswith("chr") else f"chr{chromosome}"
                chrom_file = self._find_chrom_file_in_dir(alt_chrom)
                if not chrom_file:
                    raise ValueError(f"Chromosome {chromosome} not found in directory {self.genome_path}")

            info = self._get_chrom_info_from_file(chrom_file)
            if not info:
                raise ValueError(f"Failed to read FASTA info from {chrom_file}")
            filepath = info["filepath"]
        else:
            info = self.index.get(chromosome)
            if not info:
                # Try with/without 'chr' prefix
                alt_chrom = chromosome.replace("chr", "") if chromosome.lower().startswith("chr") else f"chr{chromosome}"
                info = self.index.get(alt_chrom)
                if not info:
                    raise ValueError(f"Chromosome {chromosome} not found in genome index of {self.genome_path}")
            filepath = self.genome_path

        offset = info["offset"]
        line_len = info["line_len"]
        line_bytes = info["line_bytes"]
        total_len = info["total_len"]

        if line_len is None or line_len <= 0 or line_bytes is None or line_bytes <= 0:
            raise ValueError(f"Invalid FASTA sequence layout parameters for chromosome {chromosome}")

        # Clamp starting coordinates and compute left padding
        if position < 0:
            pad_left = abs(position)
            length = length - pad_left
            position = 0
            if length <= 0:
                return "N" * (pad_left + length)
        else:
            pad_left = 0

        if position >= total_len:
            return "N" * (pad_left + length)

        # Clamp ending coordinates and compute right padding
        if position + length > total_len:
            pad_right = (position + length) - total_len
            length = total_len - position
        else:
            pad_right = 0

        if length <= 0:
            return "N" * pad_left + "N" * pad_right

        # Compute file offsets for the chunk of sequence
        line_index = position // line_len
        char_in_line = position % line_len
        start_byte = offset + line_index * line_bytes + char_in_line

        end_pos = position + length
        end_line_index = (end_pos - 1) // line_len
        end_char_in_line = (end_pos - 1) % line_len
        end_byte = offset + end_line_index * line_bytes + end_char_in_line + 1

        bytes_to_read = end_byte - start_byte
        if bytes_to_read <= 0:
            return "N" * pad_left + "N" * pad_right

        with open(filepath, "rb") as f:
            f.seek(start_byte)
            raw_data = f.read(bytes_to_read)

        decoded = raw_data.decode("utf-8", errors="ignore")
        cleaned = decoded.replace("\n", "").replace("\r", "")
        seq = cleaned[:length]

        return "N" * pad_left + seq + "N" * pad_right


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    complement = {
        "A": "T", "T": "A", "C": "G", "G": "C", "N": "N",
        "a": "t", "t": "a", "c": "g", "g": "c", "n": "n"
    }
    return "".join(complement.get(base, base) for base in reversed(seq))
