import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import os
from typing import Optional
from DCNN_BDDQNMSA.base_net import MSABranchingNet
import config
from model_utils import ReplayBuffer


class DQNAgent:
    def __init__(self, n_sequences: int, vocab_size: int, 
                 lr: float = config.ALPHA, 
                 gamma: float = config.GAMMA, 
                 epsilon_start: float = config.EPSILON):
        self.n_sequences: int = n_sequences
        self.gamma: float = gamma
        self.epsilon: float = epsilon_start
        self.epsilon_decay: float = config.EPSILON_DECAY
        self.epsilon_min: float = 0.05
        self.device: torch.device = config.DEVICE
        
        self.policy_net: MSABranchingNet = MSABranchingNet(n_sequences, vocab_size).to(self.device)
        self.target_net: MSABranchingNet = MSABranchingNet(n_sequences, vocab_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer: optim.Adam = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory: ReplayBuffer = ReplayBuffer(capacity=config.REPLAY_MEMORY_SIZE)

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Epsilon-greedy che restituisce un vettore di azioni [N]."""
        if random.random() < self.epsilon:
            # Generates a random binary tensor.
            return torch.randint(0, 2, (self.n_sequences,), device=self.device)
        
        with torch.no_grad():
            state_input = state.unsqueeze(0).to(self.device) # (1, N, 30)
            q_values = self.policy_net(state_input)          # (1, N, 2)
            
            # q_values is a (1, N, 2) tensor (batch_size, N, [base, gap])
            # .argmax(dim=2) produces a (1, N) tensor
            # .squeeze(0) removes the batch dimension, making an (N) tensor
            actions = q_values.argmax(dim=2).squeeze(0)
            return actions

    def store_transition(self, s: torch.Tensor, a: int, r: float, s_next: torch.Tensor, done: bool) -> None:
        """Invia l'esperienza al Replay Buffer."""
        self.memory.push(s, a, r, s_next, done)

    def update_epsilon(self) -> None:
        """Riduce il fattore di esplorazione."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def learn(self, batch_size: int = 64) -> Optional[tuple]:
        if len(self.memory) < batch_size:
            return None

        # Replay Buffer Sampling
        # states: (B, N, 30), actions: (B, N), rewards: (B, 1), next_states: (B, N, 30), dones: (B, 1)
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size, self.device)

        q_out = self.policy_net(states)
        current_q = q_out.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        # current_q (B, N) -> Q_i values for N branches

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=2, keepdim=True) # (B, N, 1)
            max_next_q = self.target_net(next_states).gather(2, next_actions).squeeze(-1) # (B, N)
            
            # Bellman eq
            target_q = rewards + (self.gamma * max_next_q * (~dones)) # (B, N)

            # debug values
            v_val = q_out.mean().item()
            a_val = (q_out.max(dim=2)[0] - q_out.min(dim=2)[0]).mean().item()

        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()

        total_norm = 0.0
        for p in self.policy_net.parameters():
            if p.grad is not None:
                param_norm = torch.linalg.vector_norm(p.grad, ord=2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
    
        return loss.item(), total_norm, current_q.mean().item(), v_val, a_val


    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename: str, path:str = config.MODEL_WEIGHTS_PATH) -> None:
        """
        Save the model weights to a file.

        Parameters:
        -----------
        - filename (str): Name of the file (without extension).
        - path (str, optional): Directory where the model will be saved (default: config.DPAMSA_WEIGHTS_PATH).

        The model is saved as a `.pth` file, which can later be loaded using the `load` function.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        full_path = os.path.join(path, filename)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, full_path)
        print(f"Modello salvato in: {full_path}")

    def load(self, filename: str, path:str = config.DPAMSA_WEIGHTS_PATH) -> None:
        """
        Load the model weights from a file.  
        Parameters:
        -----------
        - filename (str): Name of the file (without extension).
        - path (str, optional): Directory from where the model will be loaded (default: config.DPAMSA_WEIGHTS_PATH).

        The function loads the weights into the `eval_net` but does not update `target_net`.
        """
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            checkpoint = torch.load(full_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            print(f"Modello caricato da: {full_path}")
        else:
            print(f"Nessun file trovato al percorso: {full_path}")