import os, re
import torch
from typing import Iterator, Union, Optional
from .encoding import SequenceEncoder

class FastaContent:
    """
    Represents the content of a .fasta file.
    """
    def __init__(self, path: str, encoder: SequenceEncoder):
        self.__path = path
        self.__encoder = encoder
        self.__sequences: Optional[list[str]] = None
        self.__tensor: Optional[torch.Tensor] = None
        self.__name = os.path.basename(path)

    def __ensure_loaded(self):
        if self.__sequences is None:
            from utils import parse_fasta_to_sequences
            with open(self.__path, 'r') as file:
                self.__sequences = parse_fasta_to_sequences(file.read())
            # Usa l'encoder iniettato per generare il tensore
            self.__tensor = self.__encoder.encode(self.__sequences)

    @property
    def tensor(self) -> torch.Tensor:
        self.__ensure_loaded()
        return self.__tensor

    @property
    def num_sequences(self) -> int:
        self.__ensure_loaded()
        return len(self.__sequences)

    @property
    def sequence_length(self) -> int:
        self.__ensure_loaded()
        return len(self.__sequences[0]) if self.__sequences else 0

    @property
    def name(self) -> str:
        return self.__name
    
    @property
    def path(self) -> str:
        return self.path



import os, re
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Iterator, Optional

# Questa rimane la classe base per il singolo elemento
class _FastaItem(Dataset):
    def __init__(self, paths: list[str], encoder):
        self.paths = paths
        self.encoder = encoder

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # Carica e codifica il singolo file
        return FastaContent(self.paths[i], self.encoder)

class FastaDataset:
    """
    Dataset per la gestione e il caricamento di file FASTA da una directory.

    Questa classe incapsula la logica per scansionare una cartella alla ricerca di file .fasta,
    ordinarli naturalmente e caricarli. Supporta l'iterazione efficiente tramite un DataLoader
    interno che utilizza multiprocessing per il caricamento asincrono dei file.

    Args:
        folder_path (str): Percorso della directory contenente i file FASTA.
        encoder (SequenceEncoder): Oggetto responsabile della codifica delle sequenze in tensori.
        num_workers (int, optional): Numero di processi worker per il caricamento dati parallelo. Default: 4.
        prefetch_factor (int, optional): Numero di campioni caricati in anticipo da ciascun worker. Default: 10.
    """
    def __init__(self, folder_path: str, encoder, num_workers: int = 4, prefetch_factor: int = 10):
        self.__path = folder_path
        self.__encoder = encoder
        self.__num_workers = num_workers
        self.__prefetch_factor = prefetch_factor
        
        # 1. Recupero e ordinamento path
        self.__fasta_paths = sorted([
            os.path.join(self.__path, f) for f in os.listdir(self.__path) 
            if f.endswith(".fasta")
        ], key=self.__natural_sort_key)

        # 2. Inizializzazione dell'oggetto Dataset interno
        self.__internal_ds = _FastaItem(self.__fasta_paths, self.__encoder)

    def __natural_sort_key(self, path: str):
        match = re.search(r"(\d+)", os.path.basename(path))
        return int(match.group(1)) if match else path

    def __len__(self) -> int:
        return len(self.__fasta_paths)

    def __iter__(self) -> Iterator[FastaContent]:
        """
        Crea un DataLoader temporaneo per l'iterazione.
        Grazie a num_workers > 0, i file vengono caricati in background.
        """
        loader = DataLoader(
            self.__internal_ds,
            batch_size=1,
            num_workers=self.__num_workers,
            prefetch_factor=self.__prefetch_factor,
            shuffle=False,
            # Necessario per restituire oggetti complessi come FastaContent
            collate_fn=lambda x: x[0] 
        )
        for fasta in loader:
            yield fasta

    def __getitem__(self, index: int) -> FastaContent:
        """
        Accesso diretto per indice. 
        Nota: Questo bypassa il prefetching (accesso sincrono).
        """
        if isinstance(index, slice):
            raise NotImplementedError("Slicing is not supported for this dataset.")
        
        return self.__internal_ds[index]