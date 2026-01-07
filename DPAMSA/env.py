import copy
from itertools import combinations
import numpy as np
import platform
import tkinter as tk
import tkinter.font as tf

import config

"""
Multiple Sequence Alignment (MSA) Environment

This script defines an environment for Multiple Sequence Alignment (MSA). 
It converts DNA sequences into numerical format, manages alignment states, 
computes scores based on matches, mismatches, and gaps, and visualizes alignments using Tkinter.

Key Features:
- Converts DNA sequences into numerical format for processing.
- Implements an alignment environment for reinforcement learning.
- Computes alignment scores based on predefined rewards and penalties.
- Provides visualization using Tkinter (Windows only).

Author: https://github.com/ZhangLab312/DPAMSA
"""

# Define colors for visualization
colors = ["#FFFFFF", "#5CB85C", "#5BC0DE", "#F0AD4E", "#D9534F", "#808080"]

# Mapping of nucleotides to numerical values
nucleotides_map = {
    'A': 1, 'a': 1,
    'T': 2, 't': 2,
    'C': 3, 'c': 3,
    'G': 4, 'g': 4,
    '-': 5,
    'N': 6, 'n': 6,
}
nucleotides = ['A', 'T', 'C', 'G', '-', 'N']


class Environment:
    """
    Environment for Multiple Sequence Alignment (MSA).

    This class processes DNA sequences, manages alignment states, computes alignment scores,
    and provides reinforcement learning interactions.

    Parameters:
    -----------
    - data (list): List of DNA sequences.
    - nucleotide_size (int, optional): Size of nucleotide representation in GUI.
    - text_size (int, optional): Font size for nucleotide labels.
    - show_nucleotide_name (bool, optional): Whether to display nucleotide names.
    - convert_data (bool, optional): Whether to convert sequences into numerical format.

    Attributes:
    ----------
    - data: List of DNA sequences converted into numerical format.
    - row: Number of sequences.
    - max_len: Length of the longest sequence.
    - aligned: List storing aligned sequences.
    - not_aligned: List storing unaligned sequences.
    - action_number: Number of possible actions (based on sequence combinations).
    - max_reward: Maximum possible alignment score.
    - GUI elements for Tkinter visualization (Windows only).
    """

    def __init__(self, data,
                 nucleotide_size=50, text_size=25,
                 show_nucleotide_name=True,
                 convert_data=True):
        # Convert DNA sequences to numerical format if required
        if convert_data:
            self.data = [[nucleotides_map[char] for char in seq] for seq in data]
        else:
            self.data = data

        self.row = len(data)
        self.max_len = max([len(data[i]) for i in range(len(data))])
        self.show_nucleotide_name = show_nucleotide_name
        self.nucleotide_size = nucleotide_size
        self.max_window_width = 1800
        self.text_size = text_size

        self.action_number = 2 ** self.row - 1
        self.max_reward = self.row * (self.row - 1) / 2 * config.MATCH_REWARD

        # Initialize alignment states
        self.aligned = [[] for _ in range(self.row)]
        self.not_aligned = copy.deepcopy(self.data)

        # Initialize Tkinter GUI (Windows only)
        if platform.system() == "Windows":
            self.window = tk.Tk()
            self.__init_size()
            self.__init_window()
            self.__init_canvas()

    def __action_combination(self):
        """Generate all possible action combinations for sequence alignment."""
        res = []
        for i in range(self.row + 1):
            combs = list(combinations(range(self.row), i))
            for j in combs:
                a = np.zeros(self.row)
                for k in j:
                    a[k] = 1
                res.append(a)

        res.pop()  # Remove last empty action
        return res

    def __init_size(self):
        """Initialize GUI window size based on sequence length."""
        self.window_default_width = (self.max_len + 2) * self.nucleotide_size if \
            (self.max_len + 2) * self.nucleotide_size < self.max_window_width else self.max_window_width
        self.window_default_height = self.nucleotide_size * (2 * self.row + 2) + 40
        self.nucleotide_font = tf.Font(family="bold", size=self.text_size * 2 // 3, weight=tf.BOLD)

    def __init_window(self):
        """Initialize Tkinter main window."""
        self.window.maxsize(self.window_default_width, self.window_default_height)
        self.window.minsize(self.window_default_width, self.window_default_height)
        self.window.title("Multiple Sequence Alignment")

    def __init_canvas(self):
        """Initialize Tkinter canvas for sequence visualization."""
        self.frame = tk.Frame(self.window, width=self.window_default_width,
                              height=self.window_default_height)
        self.frame.pack()

        self.canvas = tk.Canvas(self.frame, width=self.nucleotide_size * (self.max_len + 1),
                                height=self.nucleotide_size * (self.row + 1),
                                scrollregion=(
                                    0, 0, self.nucleotide_size * (len(self.aligned[0]) + 1),
                                    self.nucleotide_size * (self.row + 1)))

        self.scroll = tk.Scrollbar(self.frame, orient="horizontal", width=20)
        self.scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.scroll.config(command=self.canvas.xview)
        self.canvas.config(xscrollcommand=self.scroll.set, width=self.max_window_width,
                           height=self.window_default_height)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def __get_current_state(self):
        """
        Get the current alignment state as a list of nucleotide indices.

        Returns:
        --------
        - list: Flattened representation of the current state.
        """
        state = []
        for i in range(self.row):
            state.extend((self.not_aligned[i][j] if j < len(self.not_aligned[i]) else 5)
                         for j in range(len(self.not_aligned[i]) + 1))

        state.extend([0 for _ in range(self.row * (self.max_len + 1) - len(state))])
        return state

    def __calc_reward(self):
        """
        Calculate the alignment reward based on matches, mismatches, and gaps.

        Returns:
        --------
        - int: Computed reward for the current alignment step.
        """
        score = 0
        tail = len(self.aligned[0]) - 1  # Last column of alignment
        for j in range(self.row):
            for k in range(j + 1, self.row):
                if self.aligned[j][tail] == 5 or self.aligned[k][tail] == 5:
                    score += config.GAP_PENALTY  # Penalty for gaps
                elif self.aligned[j][tail] == self.aligned[k][tail]:
                    score += config.MATCH_REWARD  # Reward for matches
                elif self.aligned[j][tail] != self.aligned[k][tail]:
                    score += config.MISMATCH_PENALTY  # Penalty for mismatches

        return score

    def __show_alignment(self):
        """
        Display the current alignment using Tkinter.

        This function visually represents the aligned and unaligned sequences using colored rectangles.
        """
        self.canvas.delete(tk.ALL)
        rx_start = self.nucleotide_size // 2
        ry_start = self.nucleotide_size // 2
        nx_start = self.nucleotide_size
        ny_start = self.nucleotide_size

        # Display aligned sequences
        for i in range(self.row):
            for j in range(len(self.aligned[i])):
                self.canvas.create_rectangle(j * self.nucleotide_size + rx_start,
                                             i * self.nucleotide_size + ry_start,
                                             (j + 1) * self.nucleotide_size + rx_start,
                                             (i + 1) * self.nucleotide_size + ry_start,
                                             fill=colors[self.aligned[i][j]], outline="#757575")
                if self.show_nucleotide_name:
                    self.canvas.create_text(j * self.nucleotide_size + nx_start,
                                            i * self.nucleotide_size + ny_start,
                                            text=nucleotides[self.aligned[i][j] - 1],
                                            font=self.nucleotide_font,
                                            fill="white")

        ry_start += (self.row + 1) * self.nucleotide_size
        ny_start += (self.row + 1) * self.nucleotide_size
        for i in range(self.row):
            for j in range(len(self.not_aligned[i])):
                self.canvas.create_rectangle(j * self.nucleotide_size + rx_start,
                                             i * self.nucleotide_size + ry_start,
                                             (j + 1) * self.nucleotide_size + rx_start,
                                             (i + 1) * self.nucleotide_size + ry_start,
                                             fill=colors[self.not_aligned[i][j]], outline="#757575")
                if self.show_nucleotide_name:
                    self.canvas.create_text(j * self.nucleotide_size + nx_start,
                                            i * self.nucleotide_size + ny_start,
                                            text=nucleotides[self.not_aligned[i][j] - 1],
                                            font=self.nucleotide_font,
                                            fill="white")

        scroll_width = len(self.aligned[0]) if len(self.aligned[0]) > len(self.not_aligned[0]) else \
            len(self.not_aligned[0])
        self.canvas['scrollregion'] = (0, 0, self.nucleotide_size * (scroll_width + 1),
                                       self.nucleotide_size * (self.row + 1))
        self.window.update()

    def reset(self):
        """Reset the alignment state to its initial condition."""
        self.aligned = [[] for _ in range(self.row)]
        self.not_aligned = copy.deepcopy(self.data)
        return self.__get_current_state()

    def step(self, action):
        """
        Perform a step in the alignment process based on the given action.

        Parameters:
        -----------
        - action (int): Bitmask representing the action to perform.

        Returns:
        --------
        - reward (float): Computed reward based on alignment.
        - new_state (list): Updated sequence state.
        - done (bool): Whether the alignment process is complete.
        """
        for bit in range(self.row):
            if 0 == (action >> bit) & 0x1 and 0 == len(self.not_aligned[bit]):
                return -self.max_reward, self.__get_current_state(), 0

        total_len = 0
        for bit in range(self.row):
            if 0 == (action >> bit) & 0x1:
                self.aligned[bit].append(self.not_aligned[bit][0])
                self.not_aligned[bit].pop(0)
            else:
                self.aligned[bit].append(5) # Insert gap

            total_len += len(self.not_aligned[bit])

        return self.__calc_reward(), self.__get_current_state(), 1 if total_len > 0 else 0

    def calc_score(self):
        """
        Compute the total alignment score.

        Returns:
        --------
        - int: Sum of all pairwise alignment scores.
        """
        score = 0
        for i in range(len(self.aligned[0])):
            for j in range(self.row):
                for k in range(j + 1, self.row):
                    if self.aligned[j][i] == 5 or self.aligned[k][i] == 5:
                        score += config.GAP_PENALTY
                    elif self.aligned[j][i] == self.aligned[k][i]:
                        score += config.MATCH_REWARD
                    elif self.aligned[j][i] != self.aligned[k][i]:
                        score += config.MISMATCH_PENALTY

        return score

    def calc_exact_matched(self):
        """
        Count columns in the alignment where all sequences match exactly.

        Returns:
        --------
        - int: Number of fully matched columns.
        """
        score = 0

        for i in range(len(self.aligned[0])):
            n = self.aligned[0][i]
            flag = True
            for j in range(1, self.row):
                if n != self.aligned[j][i]:
                    flag = False
                    break
            if flag:
                score += 1

        return score

    def set_alignment(self, seqs):
        """
        Set a new alignment state.

        Parameters:
        - seqs (list): List of aligned sequences (numeric or string format).
        """
        if isinstance(seqs[0][0], int):  # If sequences are numeric
            self.aligned = seqs
        else:  # Convert string sequences to numeric format
            self.aligned = [[nucleotides_map[char] for char in seq] for seq in seqs]
        self.not_aligned = [[] for _ in range(len(self.data))]

    def render(self):
        """Render the alignment using Tkinter (Windows only)."""
        if platform.system() == "Windows":
            self.__show_alignment()

    def get_alignment(self):
        """
        Retrieve the aligned sequences as a formatted string.

        Returns:
        --------
        - str: Aligned sequences in a readable format.
        """
        alignment = ""
        for seq in self.aligned:
            if isinstance(seq[0], int):  # If sequences are numeric, convert them to string format
                alignment += ''.join([nucleotides[n - 1] for n in seq]) + '\n'
            else:  # Join characters
                alignment += ''.join(seq) + '\n'
        return alignment.rstrip()

    def padding(self):
        """
        Apply padding to ensure all sequences have the same length.

        This function fills shorter sequences with gaps ('-') to match the longest sequence.
        """
        max_length = 0
        for i in range(len(self.not_aligned)):
            max_length = max(max_length, len(self.not_aligned[i]))

        for i in range(len(self.not_aligned)):
            self.aligned[i].extend(self.not_aligned[i])
            self.aligned[i].extend([5 for _ in range(max_length - len(self.not_aligned[i]))])
            self.not_aligned[i].clear()
