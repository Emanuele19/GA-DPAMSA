from pathlib import Path
from typing import Tuple
from dataset_module.data_utils import *
from .aliases import Record

class RemoveDuplicates:
    """
    Step 4: Legge le sequenze e rimuove quelle identiche, salvando i file puliti.
    """
    def __init__(self, patterns: Tuple[str, ...] = ("*.fasta", "*.fa", "*.fna", "*.cds.fasta")):
        self.patterns = patterns

    def __call__(self, input_dir: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- Removing duplicates: {input_dir} -> {output_dir} ---")

        for in_path in iter_fasta_files(input_dir, patterns=self.patterns):
            records = read_fasta(in_path)
            
            # Funzione importata da data_utils che filtra i duplicati
            out_records = make_unique_records(records)

            write_fasta(output_dir / in_path.name, out_records, width=WRAP_DEFAULT)

            print(
                f"{in_path.name}: {len(records)} -> "
                f"{len(out_records)} unique sequences"
            )