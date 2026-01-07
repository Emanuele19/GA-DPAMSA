import torch
from torch.utils.data import IterableDataset
from pathlib import Path
from typing import List, Tuple, Iterator, Optional, Callable
from .data_utils import iter_fasta_files, read_fasta

# Type aliases
Record = Tuple[str, str]

class FastaWindowDataset(IterableDataset):
    """
    An IterableDataset that reads FASTA files, applies augmentations,
    groups sequences, and splits them into fixed-length windows.
    
    Yields:
        dict: A dictionary containing:
            - 'records': List[Record] (the processed window block)
            - 'meta': dict (metadata for filename generation)
    """

    def __init__(
        self,
        input_dir: Path,
        transform: Optional[Callable] = None,
        max_seqs_per_block: int = 3,
        window_len: int = 30,
        keep_incomplete_windows: bool = True,
        drop_incomplete_groups: bool = True,
        drop_incomplete_windows_by_seqcount: bool = False,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.transform = transform
        self.max_seqs_per_block = max_seqs_per_block
        self.window_len = window_len
        self.keep_incomplete_windows = keep_incomplete_windows
        self.drop_incomplete_groups = drop_incomplete_groups
        self.drop_incomplete_windows_by_seqcount = drop_incomplete_windows_by_seqcount
        self.seed = seed

    def _chunk_sequence(self, seq: str) -> List[str]:
        """Splits a single sequence into chunks."""
        chunks: List[str] = []
        for i in range(0, len(seq), self.window_len):
            piece = seq[i : i + self.window_len]
            if len(piece) < self.window_len and not self.keep_incomplete_windows:
                break
            chunks.append(piece)
        return chunks

    def _process_group(self, group: List[Record]) -> Iterator[Tuple[List[Record], int]]:
        """
        Splits a group of sequences into windows aligned by index.
        Yields (window_records, window_index).
        """
        # 1. Split each sequence in the group into chunks
        chunked_seqs = [
            (h, self._chunk_sequence(s)) for h, s in group
        ]
        
        # 2. Determine max number of windows in this group
        max_windows = max((len(c) for _, c in chunked_seqs), default=0)

        # 3. Create windows (slices across the group)
        for w_idx in range(max_windows):
            win_records = []
            for h, chunks in chunked_seqs:
                if w_idx < len(chunks):
                    win_records.append((h, chunks[w_idx]))
            
            # Filter logic: if we strictly require N sequences per window
            if self.drop_incomplete_windows_by_seqcount and len(win_records) != self.max_seqs_per_block:
                continue
                
            if win_records:
                yield win_records, w_idx

    def __iter__(self) -> Iterator[dict]:
        if self.seed is not None:
            # Note: In multi-worker DataLoaders, you need to handle worker_init_fn 
            # to avoid identical seeds in each worker.
            import random
            random.seed(self.seed)

        # Iterate over files
        for fasta_path in iter_fasta_files(self.input_dir):
            records = read_fasta(fasta_path)

            # Apply transform (augmentation on full sequences)
            if self.transform:
                records = self.transform(records)

            # Create groups (batching logic manually applied here to keep group cohesion)
            groups = [
                records[i : i + self.max_seqs_per_block]
                for i in range(0, len(records), self.max_seqs_per_block)
            ]

            if self.drop_incomplete_groups:
                groups = [g for g in groups if len(g) == self.max_seqs_per_block]

            base_name = fasta_path.stem

            for g_idx, group in enumerate(groups):
                # Process the group into windows
                for win_records, w_idx in self._process_group(group):
                    
                    # Yield a dictionary object ready for the consumer
                    yield {
                        "records": win_records,
                        "meta": {
                            "source": base_name,
                            "group_idx": g_idx,
                            "window_idx": w_idx
                        }
                    }