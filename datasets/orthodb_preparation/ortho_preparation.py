from pathlib import Path
from collections import Counter
import json
import random
import sys, os 
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from dataset_module.data_utils import *

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = (SCRIPT_DIR / "../../datasets/fasta_files").resolve()
RAW_DIR = DATASET_ROOT / "orthodb_v12/raw"


NO_AMBIG_DIR = DATASET_ROOT / "orthodb_v12/no_ambig_imp"
UNIQUE_DIR = DATASET_ROOT / "orthodb_v12/unique_no_ambig_imp"
CUT_DIR = DATASET_ROOT / "orthodb_v12/unique_no_ambig_imp_cut"

PATTERNS = ("*.fasta", "*.fa", "*.faa", "*.fna", "*.cds.fasta")

# Cut parameters
K = 300   # threshold
H = 100   # interval


# ==========================
# Step 1 — Counting unique characters
# ==========================

def scan_symbols(fasta_dir: Path) -> Counter:
    symbols = Counter()

    for fasta_path in iter_fasta_files(fasta_dir, patterns=PATTERNS):
        records = read_fasta(fasta_path)
        for _, seq in records:
            symbols.update(seq.upper())

    symbols.pop("", None)
    
    print("Unique symbols found in sequences:")
    print(" ".join(sorted(symbols.keys())))

    print("\nCount per symbol:")

    for sym, count in symbols.most_common():
        print(f"{sym!r}: {count}")
    
    return symbols

# ==========================
# Step 2 — Resolve ambiguous bases, 
#          using the distribution of the certain ones
# ==========================

def resolve_ambiguities(input_dir: Path, output_dir: Path, symbols: Counter):
    output_dir.mkdir(parents=True, exist_ok=True)

    base_counts = {b: symbols[b] for b in "ACGT"}
    total = sum(base_counts.values())
    base_probs = {b: base_counts[b] / total for b in "ACGT"}

    Path("global_base_probs.json").write_text(
        json.dumps(base_probs, indent=2)
    )

    print(f"Base counts: {base_counts}")
    print(f"Total base counts: {total}")
    print(f"Base probabilities: {base_probs}")

    for in_path in tqdm(
            list(iter_fasta_files(
                input_dir, 
                patterns=("*.fasta", "*.fa", "*.fna"))),
        desc="Resolving ambiguous bases",
        unit="file",
    ):
        records = read_fasta(in_path)
        new_records = []

        for header, seq in records:
            new_seq = resolve_ambiguous_sequence(seq, base_probs)
            new_records.append((header, new_seq))

        write_fasta(output_dir / in_path.name, new_records, width=60)

# ==========================
# Step 3 — Check duplicates 
# ==========================

def report_duplicates(input_dir: Path):
    for path in iter_fasta_files(input_dir, patterns=PATTERNS):
        records = read_fasta(path)
        duplicates = find_duplicate_sequences(records)

        print(f"\n=== {path.name} ===")
        if duplicates:
            print(f"Found {len(duplicates)} duplicated sequences:")
            for seq, count in duplicates.items():
                print(f" - occurrences: {count}")
        else:
            print("No duplicated sequences found.")

# ==========================
# Step 4 — Remove duplicates
# ==========================

def remove_duplicates(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for in_path in iter_fasta_files(input_dir, patterns=PATTERNS):
        records = read_fasta(in_path)
        out_records = make_unique_records(records)

        write_fasta(output_dir / in_path.name, out_records)

        print(
            f"{in_path.name}: {len(records)} → "
            f"{len(out_records)} unique sequences"
        )

# ==========================
# Step 5 — Cut sequences
# ==========================

def cut_sequences(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for fin in iter_fasta_files(input_dir, patterns=PATTERNS):
        records = read_fasta(fin)
        if not records:
            continue

        new_records = []
        lengths_before = []
        lengths_after = []

        for h, s in records:
            orig_len = len(s)
            new_len = random_cut_length(orig_len, K=K, H=H)
            new_s = s[:new_len]

            new_records.append((h, new_s))
            lengths_before.append(orig_len)
            lengths_after.append(new_len)

        fout = output_dir / fin.name
        write_fasta(fout, new_records, width=WRAP_DEFAULT)

        print(
            f"{fin.name} | nseq={len(records)} | "
            f"len_before=[{min(lengths_before)}–{max(lengths_before)}] | "
            f"len_after=[{min(lengths_after)}–{max(lengths_after)}]"
        )


# ==========================
# Main pipeline
# ==========================

def main():
    print("Step 1 — Scanning symbols")
    symbols = scan_symbols(RAW_DIR)

    print("\nUnique symbols found:")
    print(" ".join(sorted(symbols.keys())))

    print("\nStep 2 — Resolving ambiguities")
    resolve_ambiguities(RAW_DIR, NO_AMBIG_DIR, symbols)

    print("\nStep 3 — Duplicate report")
    report_duplicates(NO_AMBIG_DIR)

    print("\nStep 4 — Removing duplicates")
    remove_duplicates(NO_AMBIG_DIR, UNIQUE_DIR)

    print("\nStep 5 — Cutting sequences")
    cut_sequences(UNIQUE_DIR, CUT_DIR)

    print("\nBasic cleaning completed successfully")


if __name__ == "__main__":
    main()