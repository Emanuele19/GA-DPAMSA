import random
from typing import List, Optional
from .aliases import Record

class RandomGapSubstitution:
    """
    Replace random positions with gaps (length unchanged).
    """
    def __init__(self, gap: str = "-", p: float = 0.02, max_gaps: Optional[int] = None, avoid_existing_gaps: bool = True):
        if len(gap) != 1:
            raise ValueError("gap must be a single character.")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be between 0 and 1.")
            
        self.gap = gap
        self.p = p
        self.max_gaps = max_gaps
        self.avoid_existing_gaps = avoid_existing_gaps

    def __call__(self, records: List[Record]) -> List[Record]:
        out: List[Record] = []
        for h, s in records:
            if not s:
                out.append((h, s))
                continue

            chars = list(s)
            replaced = 0
            for i, ch in enumerate(chars):
                if self.avoid_existing_gaps and ch == self.gap:
                    continue
                if random.random() < self.p:
                    chars[i] = self.gap
                    replaced += 1
                    if self.max_gaps is not None and replaced >= self.max_gaps:
                        break

            out.append((h, "".join(chars)))
        return out