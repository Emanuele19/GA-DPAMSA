"""
split_fasta_blocks.py

Split FASTA files into smaller FASTA blocks:
- up to N sequences per block (default: 3)
- sequences chunked into fixed-length windows (default: 30)

Pipeline:
1) Read FASTA files from input directory
2) Apply augmentations on FULL sequences (before chunking)
3) Group sequences (max N per group)
4) Split each group into windows
5) Write each window as a FASTA file
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple
import random

import data_utils

Record = Tuple[str, str]
AugFn = Callable[[List[Record]], List[Record]]


# Augmentations

def aug_identity(records: List[Record]) -> List[Record]:
    """
    No-op augmentation.

    Parameters
    ----------
    records : list[tuple[str, str]]
        FASTA records as (header, sequence).

    Returns
    -------
    list[tuple[str, str]]
        Same records unchanged.
    """
    return records


def compose_augs(*augs: AugFn) -> AugFn:
    """
    Compose multiple augmentations.

    Parameters
    ----------
    *augs : callable
        One or more augmentation functions, each taking and returning `list[Record]`.

    Returns
    -------
    callable
        A single augmentation that applies all augmentations in order.
    """
    def _fn(records: List[Record]) -> List[Record]:
        for aug in augs:
            records = aug(records)
        return records
    return _fn


def aug_left_gap_pad(max_shift: int, gap: str = "-") -> AugFn:
    """
    Prepend 0..max_shift gap characters to each sequence (right-shift).
    This changes which symbols fall into each fixed window after chunking.

    Parameters
    ----------
    max_shift : int
        Maximum number of gaps to add to the left of each sequence.
        Actual shift is sampled uniformly in [0, max_shift] per sequence.
    gap : str
        Gap character to prepend (default: "-"). Must be a single character.

    Returns
    -------
    callable
        Augmentation function (records -> records).
    """
    if max_shift < 0:
        raise ValueError("max_shift must be >= 0")
    if len(gap) != 1:
        raise ValueError("gap must be a single character (e.g., '-')")

    def _fn(records: List[Record]) -> List[Record]:
        out: List[Record] = []
        for h, s in records:
            k = random.randint(0, max_shift)
            out.append((h, gap * k + s))
        return out

    return _fn

import random
from typing import List, Tuple, Callable

Record = Tuple[str, str]
AugFn = Callable[[List[Record]], List[Record]]

def aug_random_gap_insertion(
    *,
    gap: str = "-",
    p: float = 0.02,
    max_run: int = 5,
    avoid_existing_gaps: bool = False,
) -> AugFn:
    """
    With probability `p`, insert a run of k gap characters BEFORE each base,
    where k is sampled uniformly in [1, max_run].

    This keeps the original base order and grows sequence length.

    Parameters
    ----------
    gap : str
        Gap character to insert (default "-"), must be a single character.
    p : float
        Probability (0..1) of inserting a gap-run before a given character.
    max_run : int
        Maximum run length. Run length k is sampled uniformly from {1..max_run}.
    avoid_existing_gaps : bool
        If True, do not insert runs before characters that are already gaps.

    Returns
    -------
    AugFn
        records -> records augmentation.
    """
    if len(gap) != 1:
        raise ValueError("gap must be a single character.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be between 0 and 1.")
    if max_run < 1:
        raise ValueError("max_run must be >= 1")

    def _fn(records: List[Record]) -> List[Record]:
        out: List[Record] = []
        for h, s in records:
            if not s:
                out.append((h, s))
                continue

            parts: List[str] = []
            for ch in s:
                if avoid_existing_gaps and ch == gap:
                    parts.append(ch)
                    continue

                if random.random() < p:
                    k = random.randint(1, max_run)
                    parts.append(gap * k)

                parts.append(ch)

            out.append((h, "".join(parts)))
        return out

    return _fn


def aug_random_gap_substitution(
    *,
    gap: str = "-",
    p: float = 0.02,
    max_gaps: Optional[int] = None,
    avoid_existing_gaps: bool = True,
) -> AugFn:
    """
    Replace random positions with gaps (length unchanged).

    Parameters
    ----------
    gap : str
        Gap character (default "-").
    p : float
        Per-position probability of turning a character into a gap (0..1).
    max_gaps : int | None
        Optional cap on number of substitutions per sequence.
    avoid_existing_gaps : bool
        If True, do not count/replace positions that are already gaps.

    Returns
    -------
    AugFn
        records -> records augmentation.
    """
    if len(gap) != 1:
        raise ValueError("gap must be a single character.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be between 0 and 1.")
    if max_gaps is not None and max_gaps < 0:
        raise ValueError("max_gaps must be >= 0 or None.")

    def _fn(records: List[Record]) -> List[Record]:
        out: List[Record] = []
        for h, s in records:
            if not s:
                out.append((h, s))
                continue

            chars = list(s)
            replaced = 0
            for i, ch in enumerate(chars):
                if avoid_existing_gaps and ch == gap:
                    continue
                if random.random() < p:
                    chars[i] = gap
                    replaced += 1
                    if max_gaps is not None and replaced >= max_gaps:
                        break

            out.append((h, "".join(chars)))
        return out

    return _fn



# Grouping and chunking

def group_records(records: List[Record], group_size: int) -> List[List[Record]]:
    """
    Split records into consecutive groups of size `group_size`.

    Parameters
    ----------
    records : list[tuple[str, str]]
        FASTA records (header, sequence).
    group_size : int
        Max number of sequences per group.

    Returns
    -------
    list[list[tuple[str, str]]]
        List of groups; last group may be smaller if not divisible.
    """
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    return [records[i:i + group_size] for i in range(0, len(records), group_size)]


def chunk_sequence(seq: str, chunk_len: int, keep_incomplete: bool) -> List[str]:
    """
    Split a sequence into chunks of size `chunk_len`.

    Parameters
    ----------
    seq : str
        Input sequence.
    chunk_len : int
        Chunk/window length.
    keep_incomplete : bool
        If True, keep the last chunk even if shorter than `chunk_len`.
        If False, drop it.

    Returns
    -------
    list[str]
        List of chunks.
    """
    if chunk_len <= 0:
        raise ValueError("chunk_len must be > 0")

    chunks: List[str] = []
    for i in range(0, len(seq), chunk_len):
        piece = seq[i:i + chunk_len]
        if len(piece) < chunk_len and not keep_incomplete:
            break
        chunks.append(piece)
    return chunks


def split_group_into_windows(
    group: List[Record],
    window_len: int,
    keep_incomplete_windows: bool,
) -> List[List[Record]]:
    """
    Build windows by taking the same chunk index from each sequence.

    Notes
    -----
    Windows may contain fewer than `len(group)` records if some sequences
    end earlier (unless you later filter them with a boolean).

    Parameters
    ----------
    group : list[tuple[str, str]]
        One group of FASTA records (header, full sequence).
    window_len : int
        Window length (e.g., 30).
    keep_incomplete_windows : bool
        Whether to keep last chunks shorter than `window_len`.

    Returns
    -------
    list[list[tuple[str, str]]]
        List of windows; each window is a list of (header, chunk).
    """
    chunked = [(h, chunk_sequence(s, window_len, keep_incomplete_windows)) for h, s in group]
    max_windows = max((len(c) for _, c in chunked), default=0)

    windows: List[List[Record]] = []
    for w in range(max_windows):
        win = [(h, c[w]) for h, c in chunked if w < len(c)]
        if win:
            windows.append(win)
    return windows


# Main processing

def process_folder(
    input_dir: Path,
    output_dir: Path,
    *,
    max_seqs_per_block: int = 3,
    window_len: int = 30,
    keep_incomplete_windows: bool = True,
    drop_incomplete_groups: bool = True,
    drop_incomplete_windows_by_seqcount: bool = False,
    augmentation: Optional[AugFn] = None,
    seed: Optional[int] = None,
    max_boards: Optional[int] = None,
):
    """
    Process all FASTA files in `input_dir` and write window blocks to `output_dir`.

    Parameters
    ----------
    input_dir : Path
        Directory containing input FASTA files.
    output_dir : Path
        Directory where output FASTA blocks are written.
    max_seqs_per_block : int
        Number of sequences per output block (group size), default 3.
    window_len : int
        Chunk length per sequence (window length), default 30.
    keep_incomplete_windows : bool
        If True, keep chunks shorter than `window_len` (e.g., sequences < 30).
        If False, drop the final chunk if shorter than `window_len`.
    drop_incomplete_groups : bool
        If True, discard groups with fewer than `max_seqs_per_block` sequences
        (e.g., last group of 1–2 sequences).
        If False, keep them and write blocks with fewer sequences.
    drop_incomplete_windows_by_seqcount : bool
        If True, only write windows that contain exactly `max_seqs_per_block`
        records (useful if you want strictly 3x30 blocks).
        If False, allow windows with fewer records (e.g., one sequence ended).
    augmentation : callable | None
        Augmentation applied to FULL sequences before chunking.
        If None, no augmentation is applied.
    seed : int | None
        RNG seed for reproducibility. If None, uses default randomness.
    max_boards : int | None
        Maximum number of output FASTA blocks (boards) to write overall.
        If None, no limit is applied.

    Returns
    -------
    None
    """
    if seed is not None:
        random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    if augmentation is None:
        augmentation = aug_identity

    total_written = 0
    for fasta_path in data_utils.iter_fasta_files(input_dir):
        records = data_utils.read_fasta(fasta_path)

        # Apply augmentation on full sequences (before chunking)
        records = augmentation(records)

        # Group sequences
        groups = group_records(records, max_seqs_per_block)
        if drop_incomplete_groups:
            groups = [g for g in groups if len(g) == max_seqs_per_block]

        base = fasta_path.stem
        written = 0

        for g_idx, group in enumerate(groups):
            windows = split_group_into_windows(
                group,
                window_len=window_len,
                keep_incomplete_windows=keep_incomplete_windows,
            )

            for w_idx, win in enumerate(windows):
                # Optional limit on blocks written
                if max_boards is not None and total_written >= max_boards:
                    print(f"[OK] {fasta_path.name}: wrote {written} blocks\n(capped by max_boards={max_boards})")
                    return  # stop processing entirely

                if drop_incomplete_windows_by_seqcount and len(win) != max_seqs_per_block:
                    continue

                out_records = [
                    (f"{h} | src={base} | grp={g_idx} | win={w_idx}", s)
                    for h, s in win
                ]

                out_name = f"{base}__grp{g_idx:03d}__win{w_idx:05d}.fasta"
                data_utils.write_fasta(output_dir / out_name, out_records, width=data_utils.WRAP_DEFAULT)
                written += 1
                total_written += 1

        print(f"[OK] {fasta_path.name}: wrote {written} blocks")


# Example usage

if __name__ == "__main__":

    DATASET_ROOT = Path("../fasta_files")

    INPUT_DIR = DATASET_ROOT / "orthodb_v12" / "unique_no_ambig_imp_cut"
    OUTPUT_DIR = DATASET_ROOT / "orthodb_v12" / "cut_boards"

    # Example augmentation: left-gap padding only (right-shift)
    aug = compose_augs(
        aug_left_gap_pad(max_shift=5, gap="-"),
        aug_random_gap_insertion(avoid_existing_gaps=True)
    )

    process_folder(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        max_seqs_per_block=3,
        window_len=30,
        keep_incomplete_windows=True,
        drop_incomplete_groups=False,
        drop_incomplete_windows_by_seqcount=True,
        augmentation=aug,
        seed=None,
        max_boards=1000,
    )
