import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import os
from typing import Optional
from DCNNMSA.base_net import MSANet
import config
from model_utils import ReplayBuffer


class DQNAgent:
    def __init__(self, n_sequences: int, vocab_size: int, 
                 lr: float = config.ALPHA, 
                 gamma: float = config.GAMMA, 
                 epsilon_start: float = config.EPSILON):
        self.n_actions: int = 2**n_sequences - 1
        self.gamma: float = gamma
        self.epsilon: float = epsilon_start
        self.epsilon_decay: float = config.EPSILON_DECAY
        self.epsilon_min: float = 0.05
        self.device: torch.device = config.DEVICE
        
        # Modelli: Policy e Target
        self.policy_net: MSANet = MSANet(n_sequences, vocab_size).to(self.device)
        self.target_net: MSANet = MSANet(n_sequences, vocab_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer: optim.Adam = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory: ReplayBuffer = ReplayBuffer(capacity=config.REPLAY_MEMORY_SIZE)

    def select_action(self, state: torch.Tensor) -> int:
        """Epsilon-greedy per bilanciare esplorazione e sfruttamento."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            # state: (N, 30) -> (1, N, 30)
            state_input = state.unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_input)
            return int(q_values.argmax().item())

    def store_transition(self, s: torch.Tensor, a: int, r: float, s_next: torch.Tensor, done: bool) -> None:
        """Invia l'esperienza al Replay Buffer."""
        self.memory.push(s, a, r, s_next, done)

    def update_epsilon(self) -> None:
        """Riduce il fattore di esplorazione."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def learn(self, batch_size: int = 64) -> Optional[float]:
        """Esegue un passo di addestramento su un batch di esperienze."""
        if len(self.memory) < batch_size:
            return None

        # 1. Campionamento pulito tramite ReplayBuffer
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size, self.device)

        # 2. Calcolo Q(s, a) attuale
        current_q = self.policy_net(states).gather(1, actions)

        # 3. Calcolo Target Q (Bellman) tramite Target Net
        # Il calcolo viene fatto con Double Deep Q-Network:
        # Viene presa l'azione migliore dalla policy e calcolato il q val
        # dalla target per evitare bias di sovrastima
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            max_next_q = self.target_net(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + (self.gamma * max_next_q * (~dones))

        # 4. Loss e Backprop
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()

        # Calcolo della norma totale dei gradienti
        # Matematicamente corrisponde a calcolare ||G||_2 = \sqrt{\sum_i g_i^2}
        # Ovvero la norma L2 di tutti i gradienti della rete.
        # Serve per dare un punteggio alla magnitudo dell'aggiornamento 
        #   che il modello vorrebbe fare prima che venga applicato il clipping.
        # Tecnicamente la funzione clip_grad_norm_ fa questo passaggio prima di
        # clippare, ma non ne restituisce il risultato quindi lo computo a mano
        total_norm = 0.0
        for p in self.policy_net.parameters():
            if p.grad is not None:
                param_norm = torch.linalg.vector_norm(p.grad, ord=2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Clipping dei gradienti
        # Scala i gradienti in modo che non superino max_norm
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item(), total_norm, current_q.mean().item()


    def update_target_network(self) -> None:
        """Sincronizza i pesi della Target Network con la Policy Network."""
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