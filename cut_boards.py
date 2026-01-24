from pathlib import Path
import torch
from torch.utils.data import DataLoader

from dataset_module.data_utils import write_fasta, WRAP_DEFAULT
from dataset_module.transforms import Compose, LeftGapPad, RandomGapInsertion
from dataset_module.transforms import RandomCutSequence, ResolveAmbiguities 
from dataset_module.transforms import RemoveDuplicates, BasicCompose, ReplaceCharacters
from dataset_module import FastaWindowDataset

import os
import h5py
import numpy as np
from Bio import SeqIO
from tqdm import tqdm

import config

from model_utils import setup_logger
logger = setup_logger()


def fasta_to_hdf5(input_folder, output_file, N, W):
    encoding_map = config.NUCLEOTIDE_ENCODING
    
    fasta_files = [f for f in os.listdir(input_folder) if f.endswith('.fasta')]
    if not fasta_files:
        logger.info(f"No .fasta file found in {input_folder}")
        return

    M = len(fasta_files)

    logger.info(f"Found: {M} files, {N} sequences per file, sequence length {W}")

    # HDF5 file definition
    with h5py.File(output_file, 'w') as hf:
        # Making chunked file for reading speedup.
        # A chunk of shape (1, N, W) means that a single random access will return an NxW window,
        # just like reading a single old .fasta file with N sequences each of W nucleotides.
        dset = hf.create_dataset(
            'alignments', 
            shape=(M, N, W), 
            dtype='uint8', 
            chunks=(1, N, W)
        )

        # Writing
        for idx, filename in enumerate(tqdm(fasta_files, desc="Writing to HDF5...")):
            file_path = os.path.join(input_folder, filename)
            records = list(SeqIO.parse(file_path, "fasta"))
            
            # Temporary matrix for current alignment
            matrix = np.zeros((N, W), dtype='uint8')
            
            for i, record in enumerate(records):
                # Nucleotide to integer encoding
                matrix[i, :] = [encoding_map.get(char) for char in record.seq]

            # Writing to dataset
            dset[idx, :, :] = matrix

    logger.info(f"Done. HDF5 file saved to: {output_file}")


def main():
    DATASET_ROOT = Path("datasets/fasta_files")
    RAW_DIR = DATASET_ROOT / "orthodb_v12" / "raw"
    PREPARED_DIR = DATASET_ROOT / "orthodb_v12" / "unique_no_ambig_imp_cut"
    OUTPUT_DIR = DATASET_ROOT / "orthodb_v12" / "cut_boards"
    
    MAX_BOARDS = 20000

    N = 3   # Number of sequences to align
    W = 30  # Length of each sequence

    HDF5_DIR = DATASET_ROOT / "orthodb_v12" / f"hdf5_{N}x{W}.h5"

    basic_preparation = BasicCompose([
        ResolveAmbiguities(),
        RemoveDuplicates(),
        RandomCutSequence(k=300, h=100),
        ReplaceCharacters({'N': '-'})
    ])

    basic_preparation(RAW_DIR, PREPARED_DIR)
    # 1. Define Transforms
    augmentation = Compose([
        LeftGapPad(max_shift=5, gap="-"),
        RandomGapInsertion(avoid_existing_gaps=True, p=0.02)
    ])

    # 2. Instantiate Dataset
    dataset = FastaWindowDataset(
        input_dir=PREPARED_DIR,
        transform=augmentation,
        max_seqs_per_block=N,
        window_len=W,
        keep_incomplete_windows=False,
        drop_incomplete_groups=False,
        drop_incomplete_windows_by_seqcount=True,
        seed=config.SEED
    )

    # 3. Create DataLoader
    # batch_size=None disables automatic batching by the DataLoader, 
    # since our dataset already yields the "blocks" (which are effectively batches of sequences)
    # exactly as we want them saved.
    loader = DataLoader(dataset, batch_size=None, num_workers=0)

    # 4. Processing Loop
    logger.info(f"Starting processing. Output: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_written = 0

    for item in loader:
        if total_written >= MAX_BOARDS:
            logger.info(f"Reached limit of {MAX_BOARDS} boards.")
            break

        records = item['records']
        meta = item['meta']

        # Construct new headers
        # Note: 'records' comes out as a list of tuples if batch_size=None, 
        # but PyTorch collation might convert strings to lists if not careful.
        # With batch_size=None, we get the raw object yielded by __iter__.
        
        out_records = []
        for h, s in records:
            new_header = (
                f"{h} | src={meta['source']} | "
                f"grp={meta['group_idx']} | win={meta['window_idx']}"
            )
            out_records.append((new_header, s))

        # Define output filename
        out_name = (
            f"{meta['source']}__"
            f"grp{meta['group_idx']:03d}__"
            f"win{meta['window_idx']:05d}.fasta"
        )
        
        # Write to disk
        write_fasta(
            OUTPUT_DIR / out_name, 
            out_records, 
            width=WRAP_DEFAULT
        )
        
        total_written += 1
        if total_written % 100 == 0:
            logger.info(f"Written {total_written} blocks...")

    logger.info(f"Done. Total blocks written: {total_written}")

    fasta_to_hdf5(OUTPUT_DIR, HDF5_DIR, N, W)

if __name__ == "__main__":
    main()