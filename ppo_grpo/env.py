import torch
import numpy as np
from typing import List, Tuple, Dict, Union
import config


class Environment:
    """
    A stateless, one-shot Environment for Generative Multiple Sequence Alignment (MSA).

    Unlike step-based RL environments, this environment does not maintain an internal state
    transition loop. Instead, it accepts a full alignment plan (Gap Matrix) from the agent,
    reconstructs the resulting biological sequences, and computes reward metrics in a single pass.

    Key Features:
    - **Reconstruction:** Merges raw DNA sequences with predicted gap counts.
    - **Scoring:** Implements Sum-of-Pairs (SP) and Column Score (CS).
    - **Metrics:** tracks Alignment Length (AL), Exact Matches (EM), and Conservation (CS).

    Attributes:
        raw_sequences (List[List[int]]): The input batch of unaligned sequences (integers).
        mode (str): The primary objective for reward calculation ('sp' or 'cs').
    """

    def __init__(self, raw_sequences: List[List[int]], mode: str = 'sp'):
        """
        Initializes the environment with a specific problem instance (a set of sequences).

        Args:
            raw_sequences: List of integer-encoded sequences.
                           Example: [[1, 2, 3], [1, 3, 2]] where 1=A, 2=C, etc.
            mode: Reward mode. 'sp' (Sum-of-Pairs) is recommended for training stability.
        """
        self.raw_sequences = raw_sequences
        self.N = len(raw_sequences)
        self.mode = mode

        # --- Configuration & Constants ---
        # We retrieve tokens and penalties from the global config to ensure consistency.
        self.gap_token = getattr(config, 'GAP_TOKEN', 5)
        self.pad_token = getattr(config, 'PADDING_TOKEN', 0)

        # Scoring Weights
        self.match_reward = getattr(config, 'MATCH_REWARD', 4.0)
        self.mismatch_penalty = getattr(config, 'MISMATCH_PENALTY', -4.0)
        self.gap_penalty = getattr(config, 'GAP_PENALTY', -2.0)  # Slightly less severe than mismatch

    # =========================================================================
    # 1. CORE LOGIC: Reconstruction (Physics Engine)
    # =========================================================================

    def reconstruct_alignment(self, gap_matrix: List[List[int]]) -> List[List[int]]:
        """
        Deterministically builds the aligned sequences based on the agent's action.

        Logic:
            For each nucleotide in the raw sequence, the agent predicts N gaps.
            These gaps are inserted *after* the nucleotide (or before, depending on convention).
            Here, we insert gaps AFTER the current nucleotide.

        Args:
            gap_matrix: A list of lists where gap_matrix[row][i] is the number of gaps
                        to insert at position i of row r.

        Returns:
            aligned_seqs: A list of equal-length sequences (padded with gap tokens).
        """
        aligned_seqs = []
        max_len_found = 0

        for r in range(self.N):
            raw_seq = self.raw_sequences[r]
            gap_counts = gap_matrix[r]

            built_seq = []

            # Safety: Ensure we don't go out of bounds if the agent predicted too few/many actions
            limit = min(len(raw_seq), len(gap_counts))

            for i in range(limit):
                # 1. Add the actual Nucleotide
                nucleotide = raw_seq[i]
                built_seq.append(nucleotide)

                # 2. Add predicted Gaps immediately after
                num_gaps = int(gap_counts[i])
                if num_gaps > 0:
                    built_seq.extend([self.gap_token] * num_gaps)

            aligned_seqs.append(built_seq)
            max_len_found = max(max_len_found, len(built_seq))

        # --- Final Rectangularization ---
        # In MSA, all sequences must have the same length.
        # We pad shorter sequences with the GAP token (or PAD token if preferred)
        # to match the longest sequence found.
        for seq in aligned_seqs:
            if len(seq) < max_len_found:
                padding_needed = max_len_found - len(seq)
                seq.extend([self.gap_token] * padding_needed)

        return aligned_seqs

    # =========================================================================
    # 2. METRICS: Scoring (The Judge)
    # =========================================================================

    def _calc_column_sp_score(self, column: List[int]) -> float:
        """
        Calculates the Sum-of-Pairs (SP) score for a single column.

        SP Score is the sum of scores of all unique pairwise comparisons in the column.
        Formula: Sum(Score(Si, Sj)) for all i < j.
        """
        score = 0.0

        for i in range(self.N):
            for j in range(i + 1, self.N):
                c1, c2 = column[i], column[j]

                # Ignore technical padding (if distinct from gap token)
                if c1 == self.pad_token and c2 == self.pad_token:
                    continue

                # Logic: Gap vs Anything = Penalty
                if c1 == self.gap_token or c2 == self.gap_token:
                    score += self.gap_penalty
                # Logic: Match
                elif c1 == c2:
                    score += self.match_reward
                # Logic: Mismatch
                else:
                    score += self.mismatch_penalty

        return score

    def calculate_metrics(self, aligned_seqs: List[List[int]]) -> Dict[str, float]:
        """
        Computes comprehensive biological metrics for the alignment.

        Metrics:
            - SP (Sum of Pairs): General alignment quality.
            - CS (Column Score): Percentage of fully conserved columns (identical bases).
            - EM (Exact Matches): Number of columns with 100% identity.
            - AL (Alignment Length): Total length of the result.

        Returns:
            Dictionary containing 'SP', 'CS', 'EM', 'AL'.
        """
        if not aligned_seqs:
            return {"SP": 0.0, "CS": 0.0, "EM": 0, "AL": 0}

        rows = len(aligned_seqs)
        cols = len(aligned_seqs[0])

        total_sp = 0.0
        perfect_columns = 0
        exact_matches = 0

        # Iterate column by column
        for c in range(cols):
            # Extract the column c across all rows
            column = [aligned_seqs[r][c] for r in range(rows)]

            # 1. SP Score Calculation
            total_sp += self._calc_column_sp_score(column)

            # Analyze column composition
            unique_chars = set(column)
            has_gap = self.gap_token in unique_chars
            has_pad = self.pad_token in unique_chars

            # 2. CS & EM Logic
            # A column is "Conserved" or an "Exact Match" if:
            # - All characters are identical (len(unique) == 1)
            # - It is NOT a gap
            # - It is NOT technical padding
            if len(unique_chars) == 1 and not has_gap and not has_pad:
                perfect_columns += 1
                exact_matches += 1

        # Normalize CS to percentage [0-100]
        cs_percentage = (perfect_columns / cols) * 100.0 if cols > 0 else 0.0

        return {
            "AL": cols,  # Alignment Length
            "SP": total_sp,  # Total Sum of Pairs Score
            "CS": cs_percentage,  # Column Score (Percentage)
            "EM": exact_matches  # Exact Matches (Count)
        }

    # =========================================================================
    # 3. PUBLIC INTERFACE
    # =========================================================================

    def evaluate(self, gap_matrix: List[List[int]]) -> Tuple[float, List[List[int]], Dict[str, float]]:
        """
        The main entry point for the Trainer.
        Executes the full pipeline: Reconstruction -> Scoring -> Metrics.

        Args:
            gap_matrix: The action from the agent (List of Lists of gap counts).

        Returns:
            reward (float): The scalar reward signal for optimization.
            aligned_seqs (List[List[int]]): The reconstructed sequences (for debug/logging).
            metrics (Dict): Full metrics dictionary (for TensorBoard).
        """
        # 1. Reconstruct (Apply Action)
        aligned_seqs = self.reconstruct_alignment(gap_matrix)

        # 2. Calculate Stats
        metrics = self.calculate_metrics(aligned_seqs)

        # 3. Select Reward Signal
        # We usually optimize for SP, but we return other metrics for monitoring.
        reward = 0.0
        if self.mode == 'sp':
            reward = metrics["SP"]
        elif self.mode == 'cs':
            reward = metrics["CS"]
        else:
            raise ValueError(f"Unknown reward mode: {self.mode}")

        return reward, aligned_seqs, metrics