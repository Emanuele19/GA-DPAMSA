import random
from typing import List
from .aliases import Record

class LeftGapPad:
    """
    Prepend 0..max_shift gap characters to each sequence (right-shift).
    """
    def __init__(self, max_shift: int, gap: str = "-"):
        if max_shift < 0:
            raise ValueError("max_shift must be >= 0")
        if len(gap) != 1:
            raise ValueError("gap must be a single character")
        
        self.max_shift = max_shift
        self.gap = gap

    def __call__(self, records: List[Record]) -> List[Record]:
        out: List[Record] = []
        for h, s in records:
            k = random.randint(0, self.max_shift)
            out.append((h, self.gap * k + s))
        return out