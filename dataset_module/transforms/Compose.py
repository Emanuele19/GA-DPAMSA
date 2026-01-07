from typing import List
from .aliases import Record


class Compose:
    """
    Composes several transforms together.

    Args:
        transforms (list of Callable): List of transforms to compose.
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, records: List[Record]) -> List[Record]:
        for t in self.transforms:
            records = t(records)
        return records
