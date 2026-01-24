import h5py
import numpy as np
import os
from typing import Iterator, Union
import config

class MSAAlignment:
    """
    Represents a single MSA alignment loaded from HDF5 file.
    Lazy access: sequences are loaded only when requested.
    """
    def __init__(self, dataset_ref: h5py.Dataset, index: int, name: str):
        self._dataset_ref = dataset_ref
        self._index = index
        self._name = name
        # reverse mapping
        self._rev_map = {value: key for key, value in config.NUCLEOTIDE_ENCODING.items()}

    @property
    def sequences(self) -> np.ndarray:
        return self._dataset_ref[self._index]

    @property
    def sequences_as_str(self) -> list[str]:
        return ["".join(self._rev_map.get(c, 'N') for c in row) for row in self.sequences]

    @property
    def name(self) -> str:
        return self._name

    @property
    def index(self) -> int:
        return self._index


class MSADataset:
    """
    Handles an entire MSA dataset stored in an HDF5 file.
    Implements random access and iteration without loading everything into RAM.
    """
    def __init__(self, h5_path: str):
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"File {h5_path} non trovato.")
        
        self.__path = h5_path
        self.__name = os.path.basename(h5_path)
        
        # 'swmr=True' allows fast reads
        self.__h5_file = h5py.File(self.__path, 'r', swmr=True)
        self.__dset = self.__h5_file['alignments']
        
        self.__len = self.__dset.shape[0]

    def __iter__(self) -> Iterator[MSAAlignment]:
        for i in range(self.__len):
            yield self[i]

    def __getitem__(self, index: Union[int, slice]) -> Union[MSAAlignment, list[MSAAlignment]]:
        if isinstance(index, int):
            if index < 0:
                index += self.__len
            if index < 0 or index >= self.__len:
                raise IndexError("Index out of range")
            
            return MSAAlignment(self.__dset, index, f"MSA_{index}")
        
        elif isinstance(index, slice):
            start, stop, step = index.indices(self.__len)
            return [self[i] for i in range(start, stop, step)]
        
        else:
            raise TypeError("Invalid argument type")

    def __len__(self) -> int:
        return self.__len

    @property
    def name(self) -> str:
        return self.__name

    @property
    def path(self) -> str:
        return self.__path

    def close(self):
        self.__h5_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()