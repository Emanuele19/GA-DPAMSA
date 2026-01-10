import random
from pathlib import Path  # Aggiunto import esplicito
from typing import List, Optional
from dataset_module.data_utils import *
from .aliases import Record

class RandomCutSequence:
    """Taglia le sequenze con logica randomica K/H."""
    
    def __init__(self, k: int, h: int):
        self.k = k
        self.h = h

    def __call__(self, input_dir: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        patterns = ("*.fasta", "*.fa", "*.fna", "*.cds.fasta")
        
        for p in iter_fasta_files(input_dir, patterns=patterns):
            records = read_fasta(p)
            if not records:
                continue

            new_records = []
            lengths_before = []
            lengths_after = []

            for h_str, s in records:
                orig_len = len(s)
                
                # CORREZIONE QUI: Usa self.k e self.h invece di K e H
                new_len = random_cut_length(orig_len, K=self.k, H=self.h)
                
                new_s = s[:new_len]

                new_records.append((h_str, new_s))
                lengths_before.append(orig_len)
                lengths_after.append(new_len)

            fout = output_dir / p.name
            write_fasta(fout, new_records, width=WRAP_DEFAULT)

            print(
                f"{p.name} | nseq={len(records)} | "
                f"len_before=[{min(lengths_before)}–{max(lengths_before)}] | "
                f"len_after=[{min(lengths_after)}–{max(lengths_after)}]"
            )