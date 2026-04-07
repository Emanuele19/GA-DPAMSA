from pathlib import Path
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset_module.transforms import ResolveAmbiguities, RemoveDuplicates, RandomCutSequence, BasicCompose
from dataset_module import FastaWindowDataset
import config

def main():
    DATASET_ROOT = Path("datasets/fasta_files")
    
    # Cartelle della nuova pipeline
    INFERENCE_RAW = DATASET_ROOT / "orthodb_v12/inference_raw"
    INFERENCE_PREPARED = DATASET_ROOT / "orthodb_v12/inference_prepared"
    INFERENCE_READY = DATASET_ROOT / "orthodb_v12/inference_benchmark_ready"
    
    INFERENCE_READY.mkdir(parents=True, exist_ok=True)
    
    N = 3   # Numero sequenze
    W = 30  # Lunghezza sequenza
    MAX_TEST_FILES = 100 # Quanti file fasta generare al massimo

    print("1. Pre-Processing di Base per l'Inferenza...")
    basic_preparation = BasicCompose([
        ResolveAmbiguities(),
        RemoveDuplicates(),
        RandomCutSequence(k=300, h=100)
    ])
    basic_preparation(INFERENCE_RAW, INFERENCE_PREPARED)

    print("2. Creazione Dataset a finestre...")
    dataset = FastaWindowDataset(
        input_dir=INFERENCE_PREPARED,
        transform=None,
        max_seqs_per_block=N,
        window_len=W,
        keep_incomplete_windows=False,
        drop_incomplete_groups=True,
        drop_incomplete_windows_by_seqcount=True,
        seed=config.SEED
    )

    loader = DataLoader(dataset, batch_size=None, num_workers=0)

    print(f"3. Generazione dei file FASTA per il benchmark in: {INFERENCE_READY}")
    
    files_written = 0
    
    for item in tqdm(loader, desc="Creazione Fasta 3x30"):
        if files_written >= MAX_TEST_FILES:
            break
            
        records = item['records']
        # Assicuriamoci di avere esattamente N sequenze
        if len(records) < N:
            continue
            
        # Creiamo il file FASTA
        out_file = INFERENCE_READY / f"inference_test_{files_written}.fasta"
        
        with open(out_file, 'w') as f:
            for i in range(N):
                header, seq_string = records[i]
                # Scriviamo l'header originale e la sequenza tagliata a lunghezza W
                f.write(f">{header}\n")
                f.write(f"{seq_string[:W]}\n")
                
        files_written += 1

    print(f"\nFatto! {files_written} file FASTA pronti per il benchmark in: {INFERENCE_READY}")

if __name__ == "__main__":
    main()