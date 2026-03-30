from pathlib import Path
import os
import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset_module.transforms import Compose, LeftGapPad, RandomGapInsertion
from dataset_module.transforms import RandomCutSequence, ResolveAmbiguities 
from dataset_module.transforms import RemoveDuplicates, BasicCompose
from dataset_module import FastaWindowDataset

import config
from model_utils import setup_logger

logger = setup_logger()


def main():
    DATASET_ROOT = Path("datasets/fasta_files")
    RAW_DIR = DATASET_ROOT / "orthodb_v12" / "raw"
    PREPARED_DIR = DATASET_ROOT / "orthodb_v12" / "unique_no_ambig_imp_cut"
    
    # Non ci serve più questa cartella intermedia!
    # OUTPUT_DIR = DATASET_ROOT / "orthodb_v12" / "cut_boards"
    
    MAX_BOARDS = 20000
    N = 3   # Number of sequences to align
    W = 30  # Length of each sequence

    HDF5_FILE = DATASET_ROOT / "orthodb_v12" / f"hdf5_{N}x{W}.h5"
    
    # Assicurati che la directory esista
    HDF5_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 1. Pre-Processing di Base
    logger.info("Starting Basic Preparation...")
    basic_preparation = BasicCompose([
        ResolveAmbiguities(),
        RemoveDuplicates(),
        RandomCutSequence(k=300, h=100)
    ])
    basic_preparation(RAW_DIR, PREPARED_DIR)

    # 2. Instantiate Dataset
    dataset = FastaWindowDataset(
        input_dir=PREPARED_DIR,
        transform=None,
        max_seqs_per_block=N,
        window_len=W,
        keep_incomplete_windows=False,
        drop_incomplete_groups=True,
        drop_incomplete_windows_by_seqcount=True,
        seed=config.SEED
    )

    # 3. Create DataLoader
    loader = DataLoader(dataset, batch_size=None, num_workers=0)

    # 4. Direct writing to HDF5 (Bypassing intermediate FASTA files)
    logger.info(f"Starting direct conversion to HDF5: {HDF5_FILE}")
    encoding_map = config.NUCLEOTIDE_ENCODING
    
    # Stimiamo il numero di item (o usiamo MAX_BOARDS se non riusciamo a stimarlo)
    try:
        total_items = min(len(dataset), MAX_BOARDS)
    except:
        total_items = MAX_BOARDS

    with h5py.File(HDF5_FILE, 'w') as hf:
        dset = hf.create_dataset(
            'alignments', 
            shape=(total_items, N, W), 
            dtype='uint8', 
            chunks=(1, N, W)
        )

        total_written = 0
        
        # Inizializziamo una barra di caricamento
        pbar = tqdm(total=total_items, desc="Building HDF5")

        for item in loader:
            if total_written >= MAX_BOARDS:
                break

            records = item['records']
            # records è una lista di tuple (header, sequence_string)
            
            # Matrice temporanea per questo allineamento
            matrix = np.zeros((N, W), dtype='uint8')
            
            # Riempiamo la matrice
            for i, (header, seq_string) in enumerate(records):
                if i >= N: 
                    break # Sicurezza nel caso arrivino più di N sequenze
                
                # Convertiamo la stringa in interi in modo sicuro
                # Se un carattere non è nella mappa, mettiamo 0 (o il token di padding)
                int_seq = [encoding_map.get(char, 0) for char in seq_string[:W]]
                
                # Padding manuale se la sequenza è più corta di W (non dovrebbe succedere con il tuo FastaWindowDataset, ma per sicurezza)
                if len(int_seq) < W:
                    int_seq.extend([0] * (W - len(int_seq)))
                    
                matrix[i, :] = int_seq

            # Scriviamo direttamente nell'HDF5
            dset[total_written, :, :] = matrix
            
            total_written += 1
            pbar.update(1)

        pbar.close()
        
        # Ridimensioniamo il dataset HDF5 se abbiamo trovato meno file del previsto
        if total_written < total_items:
            logger.info(f"Found fewer boards than expected. Resizing HDF5 to {total_written}.")
            dset.resize((total_written, N, W))

    logger.info(f"Done. Successfully written {total_written} alignments directly to HDF5.")

if __name__ == "__main__":
    main()