import torch
from collections import deque
from typing import Tuple
import random


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s: torch.Tensor, a: int, r: float, s_next: torch.Tensor, done: bool) -> None:
        """Aggiunge una transizione alla memoria."""
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estrae un batch di esperienze e le converte in tensori pronti per il device."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Creazione dei tensori direttamente sul device per evitare passaggi CPU -> GPU multipli
        states_t = torch.stack(states).to(device)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards_t = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states_t = torch.stack(next_states).to(device)
        dones_t = torch.tensor(dones, dtype=torch.bool).to(device)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return len(self.buffer)