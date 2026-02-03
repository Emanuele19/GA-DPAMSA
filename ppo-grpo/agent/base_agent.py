import torch
import numpy as np
from abc import ABC, abstractmethod

from .interfaces import IMSAActor

class BaseMSAAgent(ABC):
    """
    Base Agent class.
    Every agent MUST have an Actor (Policy) to make decisions.
    """

    def __init__(self, actor: IMSAActor, device: str = 'cpu'):
        self.actor = actor
        self.device = device
        self.actor.to(device)

    def get_action(self, state: np.ndarray, deterministic: bool = False):
        """
        Common inference logic.
        Input: Numpy Array (Batch, Rows, Len)
        Output: Numpy Array (Batch, Rows, Len) - Actions
        """
        self.actor.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.long, device=self.device)
            # Create a simple mask: 1 where data > 0, 0 where padding (0)
            mask = (state_t != 0)

            action_t = self.actor.get_action(state_t, mask=mask, deterministic=deterministic)
            return action_t.cpu().numpy()

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass