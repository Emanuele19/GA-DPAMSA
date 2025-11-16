import os, re
from typing import Iterator, Union
from utils import parse_fasta_to_sequences

class FastaContent:
    """
    This object represents a fasta's file content.
    A .fasta file has the following format:
    
    >seq1
    ATCGGCTA
    >seq2
    TTAGCCCTA

    In the `sequences` property the content is encoded
    as a list of the sequences (strings) like:
    ['ATCGGCTA', 'TTAGCCCTA']

    """
    @staticmethod
    def __parse_fasta_file(path: str):
        with open(path, 'r') as file:
            content = file.read()

        return parse_fasta_to_sequences(content)
    
    def __init__(self, path: str):
        self.__path = path
        self.__sequences: list[str] = self.__parse_fasta_file(path)

    @property
    def path(self) -> str:
        return self.__path

    @property
    def sequences(self) -> list[str]:
        return self.__sequences
    
    @property
    def name(self) -> str:
        return os.path.basename(self.__path)

class FastaDataset:
    """
    This object represents a folder of .fasta files.
    Every file is represented by a `FastaContent` instance.
    It is possible to iterate over this object and to slice it.
    Note: this object does not contain sequences, it contains
    a list of `FastaContent`.
    """

    @staticmethod
    def __extract_number(filename: str, prefix: str) -> int:
        """
        Estrae la prima sequenza numerica dopo il prefix.
        Es: 'test12.fasta' → 12
        """
        match = re.search(rf"{re.escape(prefix)}(\d+)", filename)
        return int(match.group(1)) if match else -1


    def __init__(self, path: str, prefix: str = "test"):
        self.__path = path
        self.__prefix = prefix
        self.__name = os.path.basename(path)

        self.__fasta_paths = [
            os.path.join(self.__path, f)
            for f in os.listdir(self.__path)
            if f.startswith(self.__prefix) and f.endswith(".fasta")
        ]

        # Ordina usando il numero dopo il prefisso
        self.__fasta_paths.sort(
            key=lambda f: self.__extract_number(os.path.basename(f), self.__prefix))


    def __iter__(self) -> Iterator[FastaContent]:
        for fasta_path in self.__fasta_paths:
            yield FastaContent(fasta_path)

    def __getitem__(self, index: Union[int, slice]) -> Union[FastaContent, list[FastaContent]]:
        if isinstance(index, int):
            if index < 0:
                index += len(self.__fasta_paths)
            if index < 0 or index >= len(self.__fasta_paths):
                raise IndexError("Index out of range")
            return FastaContent(self.__fasta_paths[index])
        
        elif isinstance(index, slice):
            sliced_paths = self.__fasta_paths[index]
            return [FastaContent(path) for path in sliced_paths]

        else:
            raise TypeError("Invalid argument type")

    def __len__(self) -> int:
        return len(self.__fasta_paths)

    @property
    def name(self) -> str:
        return self.__name

    @property
    def path(self) -> str:
        return self.__path

    @property
    def fasta_files(self) -> list[str]:
        return self.__fasta_paths
