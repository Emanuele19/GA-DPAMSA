import torch
from typing import Any
from .interface import IMSAPreprocessor


class IntegerStatePreprocessor(IMSAPreprocessor):
    """
    Concrete implementation of the Preprocessor Middleware.

    This component ensures that both the Trainer and the Genetic Algorithm (GA)
    feed data to the Agent in an identical, robust format.

    It handles:
    - Extraction of raw sequences from complex objects.
    - 'Sanitization': Stripping existing alignment gaps to force the model 
      to re-align from scratch (Gap Healing).
    - Batch Padding: Creating a valid rectangular tensor for the GPU.
    """

    def __init__(self, config, device: str = 'cpu'):
        """
        Args:
            config: Configuration module containing NUCLEOTIDE_ENCODING.
            device (str): The target device ('cpu' or 'cuda') for the output tensor.
        """
        self.device = device

        # We retrieve the specific integer codes for Padding and Gaps
        # Defaulting to standard values if not found in config
        self.pad_idx = config.NUCLEOTIDES_MAP.get('P', 0)
        self.gap_idx = config.NUCLEOTIDES_MAP.get('-', 5)

    def __call__(self, batch_data: list[list[int]] | Any, sanitize: bool = True) -> torch.Tensor:
        """
        Main processing method.

        Args:
            batch_data: Raw input batch.
            sanitize (bool): 
                - If True (default): Removes existing gap tokens ('-'). 
                  This is crucial for the GA context to allow 'Destructive Re-alignment'.
                - If False: Keeps gaps as they are (rarely used in this architecture).

        Returns:
            torch.Tensor: Padded tensor (Batch, Rows, Max_Len).
        """
        # 1. Normalize Input
        # Extract raw lists of integers regardless of the input container type
        raw_batch = self._extract_raw(batch_data)

        # 2. Process Sequences (Sanitization)
        processed_batch = []
        max_len = 0

        for sub_board in raw_batch:
            cleaned_board = []
            for seq in sub_board:
                # Sanitize: Filter out the gap index if requested.
                # This treats the input sequence as a "bag of nucleotides" 
                # effectively resetting the alignment state for the model.
                if sanitize:
                    seq = [x for x in seq if x != self.gap_idx]

                cleaned_board.append(seq)

                # Track the maximum length in this batch for padding
                max_len = max(max_len, len(seq))

            processed_batch.append(cleaned_board)

        # 3. Tensor Construction (Padding)
        batch_size = len(processed_batch)
        n_rows = len(processed_batch[0])

        # Initialize the tensor filled with the Padding Index
        # This ensures that empty space is effectively "masked" later
        tensor_out = torch.full(
            (batch_size, n_rows, max_len),
            fill_value=self.pad_idx,
            dtype=torch.long
        )

        # Fill the tensor with the actual data
        for i, sub_board in enumerate(processed_batch):
            for r, seq in enumerate(sub_board):
                l = len(seq)
                # Copy the sequence into the tensor
                tensor_out[i, r, :l] = torch.tensor(seq, dtype=torch.long)

        # Move to the target hardware (GPU/CPU)
        return tensor_out.to(self.device)

    def _extract_raw(self, batch_data: Any) -> list[list[list[int]]]:
        """
        Helper method to handle different input data types.

        Supported inputs:
        1. torch.Tensor (e.g., from a pre-loaded HDF5 dataset)
        2. list of MSAAlignment objects (from your colleague's dataset module)
        3. list of lists of Integers (Raw data from GA)
        """
        # Case A: Input is already a Tensor
        # We convert it back to a list to perform list-based operations (like removing elements)
        if torch.is_tensor(batch_data):
            return batch_data.cpu().numpy().tolist()

        # Case B: Input is a list
        if isinstance(batch_data, list):
            # Check if it contains custom objects with a '.sequences' attribute
            if len(batch_data) > 0 and hasattr(batch_data[0], 'sequences'):
                return [x.sequences for x in batch_data]

            # Assume it is already a list of lists of integers
            return batch_data

        raise TypeError(f"MSAStatePreprocessor received unsupported data type: {type(batch_data)}")