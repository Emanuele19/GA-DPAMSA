import random
from pathlib import Path  # Aggiunto import esplicito
from typing import List, Optional, Tuple
from dataset_module.data_utils import *
import json
from tqdm import tqdm
from .aliases import Record

from model_utils import setup_logger
logger = setup_logger()


class ResolveAmbiguities:
    """
    Scansiona i file per calcolare la distribuzione globale di A, C, G, T
    e sostituisce le basi ambigue (N) usando queste probabilità.
    """
    def __init__(self, patterns: Tuple[str, ...] = ("*.fasta", "*.fa", "*.fna", "*.cds.fasta")):
        self.patterns = patterns

    def __call__(self, input_dir: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # --- FASE 1: Scansione Simboli ---
        logger.info(f"Scanning symbols in {input_dir}...")
        symbols = Counter()
        
        for fasta_path in iter_fasta_files(input_dir, patterns=self.patterns):
            records = read_fasta(fasta_path)
            for _, seq in records:
                symbols.update(seq.upper())

        symbols.pop("", None)
        logger.info("Unique symbols found:", " ".join(sorted(symbols.keys())))
        
        # --- FASE 2: Calcolo Probabilità ---
        base_counts = {b: symbols[b] for b in "ACGT"}
        total = sum(base_counts.values())

        if total == 0:
            logger.warning(f"Nessuna base ACGT trovata in {input_dir}. Copio i file senza modifiche o salto.")
            base_probs = {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25} 
        else:
            base_probs = {b: base_counts[b] / total for b in "ACGT"}

        logger.info(f"Base probs detected: {base_probs}")

        json_path = output_dir.parent / "global_base_probs.json"
        json_path.write_text(json.dumps(base_probs, indent=2))
        
        # --- FASE 3: Riscrittura File ---
        file_list = list(iter_fasta_files(input_dir, patterns=self.patterns))
        
        for in_path in tqdm(file_list, desc="Resolving ambiguities", unit="file"):
            records = read_fasta(in_path)
            new_records = []

            for header, seq in records:
                # Usa le probabilità calcolate
                new_seq = resolve_ambiguous_sequence(seq, base_probs)
                new_records.append((header, new_seq))

            write_fasta(output_dir / in_path.name, new_records, width=60)