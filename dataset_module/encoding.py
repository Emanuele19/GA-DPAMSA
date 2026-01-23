import torch


class SequenceEncoder:
    """
    Handles the conversion of sequences to numerical representations.

    Parameters:
    - mapping (dict[str, int]): Mapping of characters (case insensitive) to their corresponding numerical values.
    - default_value (int): Value to use for unknown characters.
    """
    def __init__(self, mapping: dict[str, int]):
        self.mapping = {k.upper(): v for k, v in mapping.items()}

    def encode(self, sequences: list[str]) -> torch.Tensor:
        numeric_data = [
            [self.mapping.get(base.upper()) for base in seq]
            for seq in sequences
        ]
        return torch.tensor(numeric_data, dtype=torch.long)

    @property
    def vocab_size(self) -> int:
        return len(self.mapping)