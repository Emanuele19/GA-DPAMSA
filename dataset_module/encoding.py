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
    

from typing import List, Dict

class SequenceDecoder:
    """
    Gestisce la conversione di tensori numerici in sequenze di nucleotidi.
    """
    def __init__(self, int_to_char: Dict[str, int]):
        self.int_to_char: Dict[int, str] = int_to_char
        
        self.pad_value = None
        for k, v in int_to_char.items():
            if v == 'P':
                self.pad_value = k
                break

    def decode_sequence(self, tensor: torch.Tensor) -> str:
        """
        Converte un singolo tensore 1D in una stringa di nucleotidi.
        """
        # Convertiamo il tensore in una lista di interi
        if isinstance(tensor, list):
            tensor = torch.tensor(tensor)
        
        indices = tensor.flatten().tolist()

        # Ricostruiamo la stringa ignorando i valori di padding
        chars = [
            self.int_to_char[idx] 
            for idx in indices 
            if idx != self.pad_value and idx in self.int_to_char
        ]
        
        return "".join(chars)

    def decode_batch(self, tensor: torch.Tensor) -> List[str]:
        """
        Data una matrice (N_seq, L), restituisce una lista di N stringhe.
        """
        return [self.decode_sequence(row) for row in tensor]