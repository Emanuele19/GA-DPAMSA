import random
from typing import List, Tuple, Optional
from .aliases import Record

class RandomGapInsertion:
    """
    With probability `p`, insert a run of k gap characters BEFORE each base.
    """
    def __init__(self, gap: str = "-", p: float = 0.02, max_run: int = 5, avoid_existing_gaps: bool = False):
        if len(gap) != 1:
            raise ValueError("gap must be a single character.")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be between 0 and 1.")
        if max_run < 1:
            raise ValueError("max_run must be >= 1")
            
        self.gap = gap
        self.p = p
        self.max_run = max_run
        self.avoid_existing_gaps = avoid_existing_gaps

    def __call__(self, records: List[Record]) -> List[Record]:
        out: List[Record] = []
        for h, s in records:
            if not s:
                out.append((h, s))
                continue

            parts: List[str] = []
            for ch in s:
                if self.avoid_existing_gaps and ch == self.gap:
                    parts.append(ch)
                    continue

                if random.random() < self.p:
                    k = random.randint(1, self.max_run)
                    parts.append(self.gap * k)

                parts.append(ch)

            out.append((h, "".join(parts)))
        return out