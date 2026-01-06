from pathlib import Path
from collections import Counter, OrderedDict
from functools import lru_cache
import random

# Default line width for FASTA output
WRAP_DEFAULT = 80

# IUPAC ambiguous nucleotide codes
AMBIGUOUS_NUCLEOTIDES = {
    "R": ["A", "G"],
    "Y": ["C", "T"],
    "S": ["G", "C"],
    "W": ["A", "T"],
    "K": ["G", "T"],
    "M": ["A", "C"],
    "B": ["C", "G", "T"],
    "D": ["A", "G", "T"],
    "H": ["A", "C", "T"],
    "V": ["A", "C", "G"],
}


def read_fasta(path: Path):
    """
    Read a FASTA file and return a list of (header, seq).

    - header is returned WITHOUT the leading '>'
    - spaces and tabs in sequence lines are removed
    """
    records = []
    header = None
    seq_lines = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    seq = "".join(seq_lines).replace(" ", "").replace("\t", "")
                    records.append((header, seq))
                header = line[1:].strip()  # remove ">" and strip
                seq_lines = []
            else:
                seq_lines.append(line.strip())

        if header is not None:
            seq = "".join(seq_lines).replace(" ", "").replace("\t", "")
            records.append((header, seq))

    return records


def write_fasta(path: Path, records, width: int = WRAP_DEFAULT):
    """Write a list of (header, seq) to FASTA, wrapping sequences to `width`."""
    with path.open("w", encoding="utf-8") as f:
        for h, s in records:
            f.write(f">{h}\n")
            for i in range(0, len(s), width):
                f.write(s[i:i + width] + "\n")


def iter_fasta_files(
    input_dir: Path,
    patterns=("*.fasta", "*.fa", "*.faa", "*.fna", "*.cds.fasta"),
):
    """
    Yield all FASTA-like files in `input_dir` matching the given patterns.
    Files are returned sorted and deduplicated.
    """
    files = []
    for pat in patterns:
        files.extend(input_dir.glob(pat))
    for p in sorted(set(files)):
        yield p


def compute_base_distribution(seqs):
    """
    Compute the distribution of A, C, G, T across all sequences.
    Only non-ambiguous bases (A, C, G, T) are considered.
    """
    counts = Counter()
    for s in seqs:
        for base in s.upper():
            if base in ("A", "C", "G", "T"):
                counts[base] += 1
    total = sum(counts.values())
    if total == 0:
        # If there are no defined bases, use a uniform distribution
        return {b: 1/4 for b in "ACGT"}
    return {b: counts[b] / total for b in "ACGT"}


@lru_cache(maxsize=None)
def _weights_for_symbol(symbol, base_probs_tuple):
    """
    Internal helper:
    Given an ambiguous symbol and the base probabilities (as an ordered tuple),
    return (list_of_possible_bases, list_of_normalized_weights).
    """
    base_order = ("A", "C", "G", "T")
    base_probs = dict(zip(base_order, base_probs_tuple))

    possible = AMBIGUOUS_NUCLEOTIDES[symbol]
    weights = [base_probs[b] for b in possible]
    s = sum(weights)
    if s == 0:
        weights = [1/len(possible)] * len(possible)
    else:
        weights = [w / s for w in weights]
    return possible, weights


def resolve_ambiguous_sequence(seq: str, base_probs):
    """
    Replace IUPAC ambiguous bases with random bases among the possible ones.
    The relative probability is based on the global A, C, G, T distribution.
    """
    base_order = ("A", "C", "G", "T")
    base_probs_tuple = tuple(base_probs[b] for b in base_order)

    new_chars = []
    for ch in seq:
        up = ch.upper()
        if up in AMBIGUOUS_NUCLEOTIDES:
            bases, weights = _weights_for_symbol(up, base_probs_tuple)
            chosen = random.choices(bases, weights=weights, k=1)[0]
            # preserve original case
            new_chars.append(chosen if ch.isupper() else chosen.lower())
        else:
            new_chars.append(ch)
    return "".join(new_chars)


def random_cut_length(orig_len: int, K: int, H: int) -> int:
    """
    Return a random length in [K-H, K], adjusted so it does not
    exceed orig_len and is at least 1. If orig_len <= K, no cut is done.
    """
    if orig_len <= K:
        return orig_len  # no cut

    min_len = max(1, K - H)
    max_len = min(K, orig_len)
    if min_len > max_len:
        min_len = max_len
    return random.randint(min_len, max_len)


def make_unique_records(records):
    """
    Given a list of (header, seq), return a new list where
    duplicate sequences are removed, keeping the first header.
    """
    unique = OrderedDict()
    for header, seq in records:
        if seq not in unique:
            unique[seq] = header
    return [(header, seq) for seq, header in unique.items()]


def find_duplicate_sequences(records):
    """
    Given a list of (header, seq), return a dict {seq: count}
    including only sequences that appear more than once.
    """
    seqs = [seq for _, seq in records]
    counter = Counter(seqs)
    return {seq: count for seq, count in counter.items() if count > 1}
