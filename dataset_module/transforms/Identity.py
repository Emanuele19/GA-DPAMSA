from typing import List
from .aliases import Record


class Identity:
    """
    No-op augmentation. Returns records unchanged.
    """
    def __call__(self, records: List[Record]) -> List[Record]:
        return records
    