from pathlib import Path
from typing import Tuple
from dataset_module.data_utils import *
from .aliases import Record


class ReportDuplicates:
    """
    Step 3: Analizza i file e riporta le sequenze duplicate a video.
    Non modifica i file e non scrive output (read-only).
    """
    def __init__(self, patterns: Tuple[str, ...] = ("*.fasta", "*.fa", "*.fna", "*.cds.fasta")):
        self.patterns = patterns

    def __call__(self, input_dir: Path, output_dir: Path = None):
        """
        L'output_dir è opzionale qui perché questo step è solo di controllo.
        """
        print(f"--- Checking duplicates in: {input_dir} ---")
        
        for path in iter_fasta_files(input_dir, patterns=self.patterns):
            records = read_fasta(path)
            duplicates = find_duplicate_sequences(records)

            print(f"\n=== {path.name} ===")
            if duplicates:
                print(f"Found {len(duplicates)} duplicated sequences:")
                for seq, count in duplicates.items():
                    print(f" - occurrences: {count}")
            else:
                print("No duplicated sequences found.")