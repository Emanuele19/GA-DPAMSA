from abc import ABC
import numpy as np
import os
import random
import torch
import torch.nn as nn

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
        #self.l3 = nn.Linear(512, action_number)
        #self.f3 = nn.Tanh()
        #duelling DQN structure can be added here if needed
        self.value_head = nn.Linear(512, 1)
        self.adv_head = nn.Linear(512, action_number)
        self.out_act = nn.Tanh()

        # Mask function to ignore certain values
        self.mask = lambda x, y: (x != y).unsqueeze(-2)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.encoder(x, self.mask(x, 0))
        # x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.f1(self.l1(x))
        x = self.f2(self.l2(x))
        value = self.value_head(x)
        adv = self.adv_head(x)
        #x = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        #x = torch.mul(x, self.max_value) # Scale output by max value
        #
        #return x

        q = value + adv - adv.mean(dim=-1, keepdim=True)
        q = self.out_act(q)
        q = torch.mul(q, self.max_value)  # Scale output by max value
        return q
    


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
            #action_val = self.eval_net.forward(torch.LongTensor(state).unsqueeze_(0).to(config.DEVICE))
            #action = torch.argmax(action_val, 1).cpu().data.numpy()[0]  # Greedy action
            with torch.no_grad():
                action_val = self.eval_net.forward(torch.LongTensor(state).unsqueeze_(0).to(config.DEVICE))
                action = torch.argmax(action_val, 1).cpu().data.numpy()[0]  # Greedy action

        return action

    def predict(self, state):
        """Predict the best action given a state."""
        action_val = self.eval_net.forward(torch.LongTensor(state).unsqueeze_(0).to(config.DEVICE))
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
        #q_next = self.target_net(batch_next_state).max(1)[0].to(config.DEVICE).detach_()
        with torch.no_grad():
            # Double DQN target: action selection with eval_net, value with target_net
            next_action = self.eval_net(batch_next_state).argmax(1, keepdin= True)
            q_next = self.target_net(batch_next_state).gather(1, next_action).squeeze_(1)
            
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
