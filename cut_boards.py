from pathlib import Path
import torch
from torch.utils.data import DataLoader

from dataset_module.data_utils import write_fasta, WRAP_DEFAULT
from dataset_module.transforms import Compose, LeftGapPad, RandomGapInsertion
from dataset_module import FastaWindowDataset

def main():
    DATASET_ROOT = Path("datasets/fasta_files")
    INPUT_DIR = DATASET_ROOT / "orthodb_v12" / "unique_no_ambig_imp_cut"
    OUTPUT_DIR = DATASET_ROOT / "orthodb_v12" / "cut_boards"
    
    MAX_BOARDS = 1000

    # 1. Define Transforms
    augmentation = Compose([
        LeftGapPad(max_shift=5, gap="-"),
        RandomGapInsertion(avoid_existing_gaps=True, p=0.02)
    ])

    # 2. Instantiate Dataset
    dataset = FastaWindowDataset(
        input_dir=INPUT_DIR,
        transform=augmentation,
        max_seqs_per_block=3,
        window_len=30,
        keep_incomplete_windows=True,
        drop_incomplete_groups=False,
        drop_incomplete_windows_by_seqcount=True,
        seed=42
    )

    # 3. Create DataLoader
    # batch_size=None disables automatic batching by the DataLoader, 
    # since our dataset already yields the "blocks" (which are effectively batches of sequences)
    # exactly as we want them saved.
    loader = DataLoader(dataset, batch_size=None, num_workers=0)

    # 4. Processing Loop
    print(f"Starting processing. Output: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_written = 0

    for item in loader:
        if total_written >= MAX_BOARDS:
            print(f"Reached limit of {MAX_BOARDS} boards.")
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
            print(f"Written {total_written} blocks...")

    print(f"Done. Total blocks written: {total_written}")

if __name__ == "__main__":
    main()