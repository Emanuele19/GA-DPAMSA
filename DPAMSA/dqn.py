from abc import ABC
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt

import config
from DPAMSA.models import Encoder
from DPAMSA.replay_memory import ReplayMemory

"""
Deep Q-Network (DQN) Implementation with PyTorch

This script implements a reinforcement learning agent using a Deep Q-Network (DQN).
It consists of a neural network (`Net`) that estimates Q-values and a `DQN` class
that manages training, experience replay, and policy updates.

Key Features:
- Uses an encoder to process input sequences.
- Implements an epsilon-greedy action selection strategy.
- Utilizes experience replay to improve sample efficiency.
- Supports model saving and loading.

Author: https://github.com/ZhangLab312/DPAMSA
"""


class Net(nn.Module):
    """
    Neural network for estimating Q-values in Deep Q-Learning.

    This network processes input sequences through an encoder and multiple
    fully connected layers with activation functions.

    Parameters:
    ----------
    - seq_num (int): Number of input sequences.
    - max_seq_len (int): Maximum sequence length.
    - action_number (int): Number of possible actions.
    - max_value (float): Maximum Q-value scaling factor.
    - d_model (int, optional): Encoder model dimension (default: 64).

    Attributes:
    -----------
    - encoder: Transformer-based encoder for input processing.
    - dropout: Dropout layer for regularization.
    - l1, l2, l3: Fully connected layers.
    - f1, f2, f3: Activation functions (LeakyReLU and Tanh).
    - mask: Function to create input masks.
    """

    def __init__(self, seq_num, max_seq_len, action_number, max_value, d_model=64):
        super(Net, self).__init__()
        self.max_value = max_value
        dim = seq_num * (max_seq_len + 1)
        self.encoder = Encoder(6, d_model, dim)
        self.dropout = nn.Dropout()
        self.l1 = nn.Linear(dim * d_model, 1028)
        self.f1 = nn.LeakyReLU()
        self.l2 = nn.Linear(1028, 512)
        self.f2 = nn.LeakyReLU()
        self.l3 = nn.Linear(512, action_number)
        self.f3 = nn.Tanh()

        # Mask function to ignore certain values
        self.mask = lambda x, y: (x != y).unsqueeze(-2)

    def forward(self, x):
        """Forward pass through the network."""
        x, attn = self.encoder(x, self.mask(x, 0))
        self.last_attention = attn
        # x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.f1(self.l1(x))
        x = self.f2(self.l2(x))
        x = self.f3(self.l3(x))
        x = torch.mul(x, self.max_value) # Scale output by max value

        return x


class DQN(ABC):
    """
    Deep Q-Network (DQN) Agent.

    This class implements a reinforcement learning agent using Deep Q-Learning.
    It maintains two neural networks (`eval_net` and `target_net`), handles
    experience replay, and updates the policy based on the Q-learning algorithm.

    Parameters:
    -----------
    - action_number (int): Number of possible actions.
    - seq_num (int): Number of sequences in input.
    - max_seq_len (int): Maximum sequence length.
    - max_value (float): Maximum value for Q-values.

    Attributes:
    -----------
    - eval_net: The network used for selecting actions.
    - target_net: The network used to compute target Q-values.
    - replay_memory: Experience replay buffer.
    - optimizer: Adam optimizer for training.
    - loss_func: Mean Squared Error loss function.
    """
    def __init__(self, action_number, seq_num, max_seq_len, max_value):
        super(DQN, self).__init__()
        self.seq_num = seq_num
        self.max_seq_len = max_seq_len
        self.action_number = action_number

        # Initialize evaluation and target networks
        self.eval_net = Net(seq_num, max_seq_len, action_number, max_value).to(config.DEVICE)
        self.target_net = Net(seq_num, max_seq_len, action_number, max_value).to(config.DEVICE)

        self.current_epsilon = config.EPSILON
        self.update_step_counter = 0
        self.epsilon_step_counter = 0

        self.replay_memory = ReplayMemory()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=config.ALPHA)
        self.loss_func = nn.MSELoss()

    def update_epsilon(self):
        """Decrease epsilon value over time for epsilon-greedy policy."""
        self.epsilon_step_counter += 1
        if self.epsilon_step_counter % config.DECREMENT_ITERATION == 0:
            self.current_epsilon = max(0, self.current_epsilon - config.DELTA)

    def select(self, state):
        """
        Select an action using an epsilon-greedy strategy.

        Parameters:
        -----------
        - state (numpy array): The current environment state.

        Returns:
        --------
        - int: Selected action index.
        """
        if random.random() <= self.current_epsilon:
            # Random policy
            action = np.random.randint(0, self.action_number) # Random action
        else:
            # Greedy policy
            action_val = self.eval_net.forward(torch.LongTensor(state).unsqueeze_(0).to(config.DEVICE))
            action = torch.argmax(action_val, 1).cpu().data.numpy()[0]  # Greedy action

        return action

    def predict(self, state, seq_lens=None):
        """Predict the best action given a state."""
        action_val = self.eval_net.forward(torch.LongTensor(state).unsqueeze_(0).to(config.DEVICE))

        # --- Interactive Attention Plot (HTML) ---
        if config.PLOT_ATTENTION:
            try:
                # Recupera la matrice di attenzione: shape (seq_len, seq_len)
                attn = self.eval_net.last_attention.detach().cpu().numpy()[0, 0]
                
                # Determina la lunghezza valida
                if seq_lens:
                    valid_len = sum(seq_lens)
                else:
                    # Fallback se seq_lens non è fornito
                    valid_len = len(state)
                    for i in range(len(state) - 1, -1, -1):
                        if state[i] != 0:
                            valid_len = i + 1
                            break
                
                # Taglia i dati alla parte valida
                attn = attn[:valid_len, :valid_len]
                seq_ints = state[:valid_len]
                
                # Mappa interi -> nucleotidi (0=Pad, 1=A, 2=T, 3=C, 4=G, 5=-, 6=N)
                vocab = ['P', 'A', 'T', 'C', 'G', '-', 'N']
                labels = [vocab[i] if 0 <= i < len(vocab) else '?' for i in seq_ints]
                
                # Prepara i dati per l'HTML
                # Ricostruisce la struttura a righe basandosi su seq_lens
                rows_html = ""
                current_idx = 0
                
                # Se seq_lens non c'è, trattiamo tutto come un'unica riga
                iter_lens = seq_lens if seq_lens else [valid_len]
                
                for length in iter_lens:
                    rows_html += '<div class="seq-row">'
                    for _ in range(length):
                        char = labels[current_idx]
                        # data-idx è l'indice globale nella matrice di attenzione
                        rows_html += f'<span class="nuc" data-idx="{current_idx}">{char}</span>'
                        current_idx += 1
                    rows_html += '</div>'

                # Serializza la matrice di attenzione per JS
                # Arrotondiamo per risparmiare spazio
                attn_json = json.dumps(np.round(attn, 4).tolist())

                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        body {{ font-family: monospace; background: #f0f0f0; padding: 20px; }}
                        .seq-row {{ display: flex; margin-bottom: 5px; }}
                        .nuc {{ 
                            width: 20px; height: 20px; display: flex; align-items: center; justify-content: center;
                            border: 1px solid #ccc; background: white; cursor: crosshair; margin-right: 1px;
                            font-weight: bold;
                        }}
                        .nuc:hover {{ border-color: #000; }}
                    </style>
                </head>
                <body>
                    <h3>Interactive Attention Map (Hover over a nucleotide)</h3>
                    <div id="grid">{rows_html}</div>
                    <script>
                        const attn = {attn_json};
                        const cells = document.querySelectorAll('.nuc');
                        
                        cells.forEach(cell => {{
                            cell.addEventListener('mouseenter', (e) => {{
                                const idx = parseInt(e.target.dataset.idx);
                                const rowAttn = attn[idx];
                                
                                // Colora tutte le celle in base all'attenzione ricevuta dalla cella corrente
                                cells.forEach((c, i) => {{
                                    const val = rowAttn[i];
                                    // Mappa valore 0-1 a colore (Bianco -> Rosso)
                                    const intensity = Math.min(1, val * 5); // Moltiplico per evidenziare valori bassi
                                    c.style.backgroundColor = `rgba(255, 0, 0, ${{intensity}})`;
                                    c.style.color = intensity > 0.5 ? 'white' : 'black';
                                }});
                            }});
                            
                            // Reset opzionale quando si esce dalla griglia
                            document.getElementById('grid').addEventListener('mouseleave', () => {{
                                cells.forEach(c => {{ c.style.backgroundColor = 'white'; c.style.color = 'black'; }});
                            }});
                        }});
                    </script>
                </body>
                </html>
                """

                save_dir = os.path.join(config.BASE_RESULTS_PATH, "attention_plots")
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, f"attn_{time.time()}.html"), "w") as f:
                    f.write(html_content)
                    
            except Exception as e:
                print(f"Warning: Could not plot attention heatmap. Error: {e}")
        # ------------------------------

        return torch.argmax(action_val, 1).cpu().data.numpy()[0]

    def update(self):
        """Update the DQN model using experience replay."""
        self.update_step_counter += 1
        if self.update_step_counter % config.UPDATE_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        if self.replay_memory.size < config.BATCH_SIZE:
            return

        # Sample batch from replay memory
        state, next_state, action, reward, done = self.replay_memory.sample(config.BATCH_SIZE)

        # Convert batch data to tensors
        batch_state = torch.LongTensor(state).to(config.DEVICE)
        batch_next_state = torch.LongTensor(next_state).to(config.DEVICE)
        batch_action = torch.LongTensor(action).to(config.DEVICE)
        batch_reward = torch.FloatTensor(reward).to(config.DEVICE)
        batch_done = torch.FloatTensor(done).to(config.DEVICE)

        # Compute Q-values and targets
        q_eval = self.eval_net(batch_state).gather(1, batch_action.unsqueeze_(-1)).squeeze_(1).to(config.DEVICE)
        q_next = self.target_net(batch_next_state).max(1)[0].to(config.DEVICE).detach_()
        q_target = batch_reward + batch_done * config.GAMMA * q_next

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()  # Return the loss value as a float

    def save(self, filename, path=config.DPAMSA_WEIGHTS_PATH):
        """
        Save the model weights to a file.

        Parameters:
        -----------
        - filename (str): Name of the file (without extension).
        - path (str, optional): Directory where the model will be saved (default: config.DPAMSA_WEIGHTS_PATH).

        The model is saved as a `.pth` file, which can later be loaded using the `load` function.
        """
        torch.save(self.eval_net.state_dict(), os.path.join(path, f"{filename}.pth"))
        print(f"\nModel weights saved as {filename}.pth in {path}\n")

    def load(self, filename, path=config.DPAMSA_WEIGHTS_PATH):
        """
        Load the model weights from a file.

        Parameters:
        -----------
        - filename (str): Name of the file (without extension).
        - path (str, optional): Directory from where the model will be loaded (default: config.DPAMSA_WEIGHTS_PATH).

        The function loads the weights into the `eval_net` but does not update `target_net`.
        """
        self.eval_net.load_state_dict(torch.load(os.path.join(path, f"{filename}.pth"),
                                                 map_location=torch.device(config.DEVICE)))
