import torch
import numpy as np

from .base_agent import BaseMSAAgent
from .interfaces import  IMSAActor, IMSACritic

class PPO_Agent(BaseMSAAgent):
    """
    Standard PPO Agent.
    Requires: Actor AND Critic.
    """

    def __init__(self,
                 actor: IMSAActor,
                 critic: IMSACritic,
                 lr: float = 1e-4,
                 device: str = 'cpu'):
        super().__init__(actor, device)
        self.critic = critic
        self.critic.to(device)

        # PPO optimizes both Actor and Critic
        # We can use one optimizer for both or two separate ones.
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])

    def get_value(self, state: np.ndarray) -> np.ndarray:
        """
        Specific to PPO: Estimates state value V(s).
        """
        self.critic.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.long, device=self.device)
            mask = self.build_mask(state_t)
            value = self.critic.get_value(state_t, mask=mask)
            return value.cpu().numpy()

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])