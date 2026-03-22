import h5py
import os
from pathlib import Path

def h5_to_fasta_converter(h5_file_path, mapper):
    """
    Converte i tensori uint8 in un dataset HDF5 'alignments' in file .fasta.
    
    Args:
        h5_file_path (str): Path del file .h5 di input.
        mapper (dict): Dizionario di mappatura {int: str} (es. {1: 'A', 2: 'C', ...})
    """
    
    h5_path = Path(h5_file_path)
    output_dir = h5_path.parent / h5_path.stem
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    try:
        with h5py.File(h5_file_path, 'r') as f:
            if 'alignments' not in f:
                print("[Error]: dataset 'alignments' not found in file")
                return

            alignments = f['alignments']
            
            for i in range(len(alignments)):
                tensor = alignments[i]
                
                fasta_path = output_dir / f"test{i}.fasta"
                
                with open(fasta_path, 'w') as fasta_file:
                    for seq_idx, row in enumerate(tensor):
                        # Conversion to nucleotide char
                        sequence_str = "".join(mapper.get(val, 'N') for val in row)
                        
                        # Fasta format
                        fasta_file.write(f">Sequence_{seq_idx + 1}\n")
                        fasta_file.write(f"{sequence_str}\n")
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1} file...")

        print(f"\Done. Output:{output_dir}")

    except Exception as e:
        print(f"Unhandled exception: {e}")

# --- ESEMPIO DI UTILIZZO ---
if __name__ == "__main__":
    from config import NUCLEOTIDE_DECODING
    import argparse

    parser = argparse.ArgumentParser(
        description="Splits an HDF5 dataset into training and test sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_file", type=str, help="Input HDF5 file path.")
    args = parser.parse_args()
    

    h5_to_fasta_converter(args.input_file, NUCLEOTIDE_DECODING)