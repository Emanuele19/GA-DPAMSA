import random

import config

"""
Replay Memory Buffer for Deep Q-Network (DQN)

This module implements a replay memory mechanism for reinforcement learning.
The buffer stores past experiences, allowing the agent to sample mini-batches 
of past transitions to stabilize learning.

Key Features:
- Implements experience replay to break temporal correlations.
- Uses a circular buffer to manage memory efficiently.
- Supports random sampling for training stability.

Author: https://github.com/ZhangLab312/DPAMSA
"""


class ReplayMemory:
    """
    Experience replay buffer for reinforcement learning.

    This class stores past experiences (state, action, reward, next_state, done)
    and allows random sampling for training.

    Attributes:
    -----------
    - storage (list): Stores experience tuples.
    - max_size (int): Maximum size of the buffer (defined in `config`).
    - size (int): Current number of stored experiences.
    - ptr (int): Pointer to track the next insertion index.
    - previous_hash (optional): Placeholder for future state tracking.
    """
    def __init__(self):
        self.storage = []
        self.max_size = config.REPLAY_MEMORY_SIZE  # Max buffer size from config
        self.size = 0  # Current memory size
        self.ptr = 0  # Pointer to track insertions
        self.previous_hash = None  # Placeholder for additional tracking (if needed)

    def push(self, data: tuple):
        """
        Add a new experience to the replay memory.

        If the buffer is full, the oldest experience is replaced using a circular buffer.

        Parameters:
        -----------
        - data (tuple): A transition tuple (state, next_state, action, reward, done).
        """
        #if len(self.storage) == self.max_size:
        #    self.storage[self.ptr-1] = data
        #    self.ptr = (self.ptr + 1) % self.max_size
        #else:
        #    self.storage.append(data)
        #    self.ptr += 1
        #    self.size += 1
        
        # Revised circular buffer logic
        # replay more stable
        if len (self.storage) == self.max_size:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)
            self.size += 1


    def sample(self, batch_size):
        """
        Sample a random batch of experiences from the replay memory.

        Parameters:
        -----------
        - batch_size (int): Number of experiences to sample.

        Returns:
        --------
        - Tuple: Batch of (states, next_states, actions, rewards, done_flags).
        """
        samples = random.sample(self.storage, batch_size)
        state, next_state, action, reward, done = [], [], [], [], []

        for i in range(batch_size):
            s, ns, a, r, d = samples[i]
            state.append(s)
            next_state.append(ns)
            action.append(a)
            reward.append(r)
            done.append(d)

        return state, next_state, action, reward, done
