import torch
from .base_agent import BaseMSAAgent
from .interfaces import IMSAActor

class GRPO_Agent(BaseMSAAgent):
    """
    Generative RPO Agent.
    Requires: Actor ONLY.
    (The 'Critic' is replaced by the Group Mean Baseline during training loop).
    """

    def __init__(self,
                 actor: IMSAActor,
                 lr: float = 1e-4,
                 device: str = 'cpu'):
        super().__init__(actor, device)

        # GRPO optimizes ONLY the Actor
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

    # Note: No 'get_value' method here. GRPO doesn't care about V(s).

    def save(self, path: str):
        torch.save({'actor': self.actor.state_dict()}, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])