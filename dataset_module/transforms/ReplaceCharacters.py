from pathlib import Path
from typing import Dict, Tuple
from dataset_module.data_utils import *
from .aliases import Record

from model_utils import setup_logger
logger = setup_logger()



class ReplaceCharacters:
    """
    Transform offline che sostituisce caratteri nelle sequenze nucleotidiche
    basandosi su un dizionario di mapping (es. {'N': '-'}).
    
    Segue il pattern di progettazione di RandomCutSequence e ResolveAmbiguities,
    operando su directory di file FASTA.
    """
    def __init__(self, mapping: Dict[str, str], patterns: Tuple[str, ...] = ("*.fasta", "*.fa", "*.fna", "*.cds.fasta")):
        self.mapping = mapping
        self.patterns = patterns
        # Pre-calcola la tabella di traduzione per performance ottimali
        self.trans_table = str.maketrans(mapping)

    def __call__(self, input_dir: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"--- Replacing characters {self.mapping}: {input_dir} -> {output_dir} ---")

        for in_path in iter_fasta_files(input_dir, patterns=self.patterns):
            records = read_fasta(in_path)
            new_records = []
            
            changed_count = 0
            for header, seq in records:
                # translate è molto veloce per sostituzioni carattere-carattere
                new_seq = seq.translate(self.trans_table)
                
                if new_seq != seq:
                    changed_count += 1
                
                new_records.append((header, new_seq))

            write_fasta(output_dir / in_path.name, new_records, width=WRAP_DEFAULT)
            
            logger.info(f"{in_path.name}: Processed {len(records)} sequences (modified {changed_count})")