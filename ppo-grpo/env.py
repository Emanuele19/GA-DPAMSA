import utils
import numpy as np
import config


class Environment:
    """
    A simplified, stateless environment for Generative MSA (One-Shot).

    This version implements the 'L-slots' logic:
    For a sequence of length L, the model predicts L integers.
    gap_matrix[i] implies: "Insert N gaps BEFORE nucleotide i".

    Trailing gaps (after the last nucleotide) are handled automatically 
    via deterministic padding to ensure rectangular alignment.
    """

    def __init__(self, raw_sequences: list[list[int]], mode: str = 'sp'):
        """
        Initialize the environment.

        Args:
            raw_sequences: A list of numerical sequences (without gaps).
                           Example: [[1, 2, 3], [1, 3, 2]]
            mode: The scoring mode ('sp' for Sum-of-Pairs, 'cs' for Column Score).
        """
        self.raw_sequences = raw_sequences
        self.num_rows = len(raw_sequences)
        self.seq_lengths = [len(seq) for seq in raw_sequences]
        self.mode = mode

        # GAP value from config (usually 5)
        self.gap_token = config.GAP_TOKEN

    def get_decision_slots(self) -> list[int]:
        """
        Returns the number of decision slots required for each sequence.

        UPDATED LOGIC:
        For a sequence of length L, we need exactly L decision slots.
        Slot[i] determines how many gaps to insert strictly BEFORE nucleotide[i].

        Returns:
            list[int]: A list containing L for each row.
        """
        return self.seq_lengths

    def reconstruct_alignment(self, gap_matrix: np.ndarray | list[list[int]]) -> list[list[int]]:
        """
        Deterministically reconstructs the alignment based on gap decisions.

        Note:
            Final alignment does not cut out gap columns.
        Args:
            gap_matrix: A matrix (Num_Rows x Max_Slots) of integers.
                        Since we use L slots, gap_matrix[r][i] corresponds exactly
                        to the gap insertion before raw_sequences[r][i].

        Returns:
            aligned_seqs: The fully aligned sequences (padded to form a rectangle).
        """
        aligned_seqs = []
        max_len_found = 0

        for r in range(self.num_rows):
            original_seq = self.raw_sequences[r]
            gap_counts = gap_matrix[r]

            built_seq = []

            # 1. Interleave gaps and nucleotides
            # We iterate exactly through the length of the original sequence.
            for i in range(len(original_seq)):
                # A. Insert gaps predicted for this position
                # (These gaps appear BEFORE the current nucleotide)
                num_gaps = int(gap_counts[i])
                if num_gaps > 0:
                    built_seq.extend([self.gap_token] * num_gaps)

                # B. Insert the actual nucleotide
                built_seq.append(original_seq[i])

            # Note: We no longer check for gaps "after" the last nucleotide.
            # That responsibility now belongs entirely to step 2 (Padding).

            aligned_seqs.append(built_seq)
            max_len_found = max(max_len_found, len(built_seq))

        # 2. Final Padding (Rectangularization)
        # Calculate the target length (Max Length) and fill the tails with gaps.
        for seq in aligned_seqs:
            if len(seq) < max_len_found:
                padding_needed = max_len_found - len(seq)
                seq.extend([self.gap_token] * padding_needed)

        return aligned_seqs

    def compute_reward(self, aligned_seqs: list[list[int]]) -> float:
        """
        Computes the alignment score (Reward) using the provided utility functions.
        """
        if not aligned_seqs:
            return 0.0

        rows = len(aligned_seqs)
        cols = len(aligned_seqs[0])

        if self.mode == 'sp':
            return utils.get_sum_of_pairs(aligned_seqs, 0, rows, 0, cols)

        elif self.mode == 'cs':
            return utils.get_column_score(aligned_seqs, 0, rows, 0, cols)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def evaluate(self, gap_matrix: np.ndarray | list[list[int]]) -> tuple[float, list[list[int]]]:
        """
        A helper method to reconstruct and score in one pass.
        Returns: tuple: (Score, Aligned_Sequences)
        """
        aligned_seqs = self.reconstruct_alignment(gap_matrix)
        score = self.compute_reward(aligned_seqs)
        return score, aligned_seqs