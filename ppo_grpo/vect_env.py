import torch
from scoring.batch_scoring import compute_cs_score, compute_sp_score


class VectorizedEnvironment:
    """
    High-Performance GPU Environment for Multiple Sequence Alignment (MSA).

    This class replaces list-based CPU manipulations with vectorized PyTorch operations.
    It treats the alignment problem as a geometric projection ("Scatter") of nucleotides
    onto a larger grid, allowing simultaneous processing of the entire batch.

    Key Concepts:
    - **Scatter vs Insert:** Instead of inserting gaps iteratively, we calculate the 
      final coordinate of every nucleotide and move them all at once.
    - **Broadcasting:** Scoring compares all rows against all rows simultaneously.
    """

    def __init__(self, config, device: torch.device):
        """
        Args:
            config: Configuration object containing token IDs and reward weights.
            device: The torch device (cuda/cpu) where tensors reside.
        """
        self.device = device
        self.config = config

        # --- Token Definitions ---
        # Must match the IntegerPreProcessor and Config
        self.gap_token = getattr(config, 'GAP_TOKEN', 5)
        self.pad_token = getattr(config, 'PADDING_TOKEN', 0)

        # --- Reward Weights ---
        self.match_reward = getattr(config, 'MATCH_REWARD', 4.0)
        self.mismatch_penalty = getattr(config, 'MISMATCH_PENALTY', -4.0)
        self.gap_penalty = getattr(config, 'GAP_PENALTY', -2.0)

    def reconstruct_alignment(self, raw_seqs: torch.Tensor, gap_counts: torch.Tensor) -> torch.Tensor:
        """
        Deterministically builds the aligned matrix using Vectorized Scatter.

        Logic:
            1. Calculate the 'Shift' for every nucleotide based on cumulative gaps.
            2. Determine the Final Position = Original Index + Shift.
            3. Create a blank Canvas filled with Gap Tokens.
            4. 'Scatter' (project) the raw nucleotides onto the Canvas at Final Positions.

        Args:
            raw_seqs: (Batch, Rows, Len) - Integer tensor of DNA residues.
            gap_counts: (Batch, Rows, Len) - Integer tensor of gaps to insert *after* each residue.

        Returns:
            aligned_tensor: (Batch, Rows, Max_Len) - The aligned sequences.
        """
        batch_size, num_rows, seq_len = raw_seqs.shape

        # 1. Calculate Shift (Cumulative Sum of Gaps)
        # We interpret gap_counts[i] as gaps BEFORE nucleotide i.
        # Therefore, the shift for nucleotide i is the sum of all gaps from index 0 up to i (inclusive).
        # Example: Gaps [2, 0, 1] -> CumSum [2, 2, 3]
        # shape: (Batch, Rows, Len)
        shifts = torch.cumsum(gap_counts, dim=2)

        # 2. Calculate Final Positions
        # Create a grid of original indices [0, 1, 2, ..., Len-1]
        original_indices = torch.arange(seq_len, device=self.device).view(1, 1, seq_len)

        # Broadcast indices across Batch and Rows
        # Final Position = Original Index + Total Gaps Before It
        final_indices = original_indices + shifts

        # 3. Create Canvas
        # Find the maximum length needed for this specific batch
        # (This is dynamic padding, much more efficient than fixed size)
        max_aligned_len = final_indices.max().item() + 1

        # Initialize canvas with GAP tokens
        aligned_tensor = torch.full(
            (batch_size, num_rows, max_aligned_len),
            self.gap_token,
            device=self.device,
            dtype=torch.long  # Must be long for embedding layers usually
        )

        # 4. Scatter (The Core Operation)
        # scatter_(dim, index, src)
        # We write 'raw_seqs' values into 'aligned_tensor' at 'final_indices'
        aligned_tensor.scatter_(2, final_indices, raw_seqs)

        # Note: 'raw_seqs' might contain PAD tokens (0). These are also scattered.
        # They will appear in the final alignment. The scoring function MUST mask them out.

        return aligned_tensor

    def evaluate_batch(
            self,
            raw_seqs: torch.Tensor,
            gap_counts: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Main entry point for the Trainer.
        Performs fully vectorized alignment and scoring on the GPU.

        Args:
            raw_seqs: (Batch, Rows, Len) - DNA Integers
            gap_counts: (Batch, Rows, Len) - Gap Action Integers

        Returns:
            rewards: (Batch,) - Tensor containing the reward for optimization.
            metrics: dict - Aggregated metrics (SP, CS, EM, AL) averaged for logging.
        """
        # 1. Alignment (GPU Scatter)
        # No CPU loops. O(1) kernel launch.
        aligned_matrix = self.reconstruct_alignment(raw_seqs, gap_counts)

        # 2. Scoring (GPU Broadcast)
        sp_scores = compute_sp_score(
            aligned_matrix,
            match_reward=self.match_reward,
            mismatch_penalty=self.mismatch_penalty,
            gap_penalty=self.gap_penalty,
            gap_token=self.gap_token,
            pad_token=self.pad_token
        )

        cs_scores = compute_cs_score(
            aligned_matrix,
            gap_token=self.gap_token,
            pad_token=self.pad_token
        )

        # 3. Aggregation for Logging
        # We calculate aux metrics for TensorBoard
        # Note: These are detached from graph as we don't backprop through metrics usually
        with torch.no_grad():
            avg_sp = sp_scores.mean().item()
            avg_cs = cs_scores.mean().item()

            # Helper for Alignment Length
            avg_al = float(aligned_matrix.shape[2])

        metrics = {
            "SP": avg_sp,
            "CS": avg_cs,
            "AL": avg_al,
            "EM": 0.0  # Placeholder
        }

        # 4. Return Reward
        # We optimize for SP score
        return sp_scores, metrics