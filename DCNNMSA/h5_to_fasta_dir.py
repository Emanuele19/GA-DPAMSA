import h5py
import os
import argparse
import numpy as np
from tqdm import tqdm
import config

ID_TO_CHAR = config.NUCLEOTIDE_DECODING

def decode_sequence(seq_array):
    """Decodes a numpy array of integers into a string."""
    # Using int(x) to correctly handle numpy types
    return "".join([ID_TO_CHAR.get(int(x), '') for x in seq_array])

def h5_to_fasta_dir(input_h5: str, output_path: str):
    """
    Extracts sequences from an HDF5 file and creates a directory containing FASTA files.
    """
    if not os.path.exists(input_h5):
        raise FileNotFoundError(f"Input file not found: {input_h5}")

    # Determine the output folder name (h5 filename without extension)
    base_name = os.path.splitext(os.path.basename(input_h5))[0]
    target_dir = os.path.join(output_path, base_name)
    
    os.makedirs(target_dir, exist_ok=True)
    print(f"Extracting sequences from {input_h5} to {target_dir}...")

    with h5py.File(input_h5, 'r') as f:
        if 'alignments' not in f:
             raise KeyError(f"Dataset 'alignments' not found in {input_h5}")
        
        data = f['alignments']
        total = data.shape[0]
        
        for i in tqdm(range(total), desc="Generating FASTA files"):
            # data[i] is a matrix of shape (N_sequences, Length)
            alignment_matrix = data[i]
            
            # Required filename: test[n].fasta
            fasta_filename = f"test{i}.fasta"
            fasta_path = os.path.join(target_dir, fasta_filename)
            
            with open(fasta_path, 'w') as fasta_file:
                num_sequences = alignment_matrix.shape[0]
                for seq_idx in range(num_sequences):
                    # Decode the sequence
                    raw_seq = alignment_matrix[seq_idx]
                    seq_str = decode_sequence(raw_seq)
                    
                    # Writing in FASTA format
                    # Header: >Sequence_1, >Sequence_2, etc.
                    fasta_file.write(f">Sequence_{seq_idx+1}\n")
                    fasta_file.write(f"{seq_str}\n")

    print(f"Extraction completed. Created {total} files in {target_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Extracts sequences from an HDF5 file and saves them as individual FASTA files in a folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_h5", type=str, help="Path to the input HDF5 file.")
    parser.add_argument("output_path", type=str, help="Path to the output directory where the folder will be created.")
    
    args = parser.parse_args()
    
    try:
        h5_to_fasta_dir(args.input_h5, args.output_path)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
