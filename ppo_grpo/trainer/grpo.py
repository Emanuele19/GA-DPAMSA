import torch
import numpy as np
from .base import BaseTrainer
from ppo_grpo.env import Environment


class GRPOTrainer(BaseTrainer):
    """
    Generative Group Relative Policy Optimization (GRPO) Trainer.

    This trainer implements a one-shot reinforcement learning loop where:
    1.  **Expansion:** Input sequences are replicated `G` times (Group Size).
    2.  **Sampling:** The Policy (Actor) predicts gap distributions for all G copies.
    3.  **Action Selection:** We sample continuous actions and decode them into integer gap counts.
    4.  **Evaluation:** The Environment reconstructs the sequences (DNA + Gaps) and computes rewards.
    5.  **Optimization:** A GRPO loss is calculated using the advantage of each sample relative to its group mean.

    Attributes:
        group_size (int): Number of parallel generations per input sequence.
        env_mode (str): Reward mode ('sp' for Sum-of-Pairs, 'cs' for Column Score).
    """

    def __init__(self, agent, group_size=8, grpo_epochs=4, clip_eps=0.2, env_mode='sp', **kwargs):
        super().__init__(agent, **kwargs)
        self.group_size = group_size
        self.grpo_epochs = grpo_epochs
        self.clip_eps = clip_eps
        self.env_mode = env_mode

    def _get_raw_sequences(
            self,
            batch_data,
            tensor_batch
    ) -> list[list[list[int]]]:
        """
        Helper to extract clean list-of-lists (without padding) from the batch.

        Args:
            batch_data: Raw data from DataLoader (list or custom object).
            tensor_batch: The processed tensor used for the network forward pass.

        Returns:
            list[list[list[int]]]: The raw DNA sequences (integers) without technical padding.
        """
        # If the dataloader yielded objects with a 'sequences' attribute, use that directly.
        if isinstance(batch_data, list) and not torch.is_tensor(batch_data):
            if hasattr(batch_data[0], 'sequences'):
                return [x.sequences for x in batch_data]
            if len(batch_data) > 0 and isinstance(batch_data[0], list) and isinstance(batch_data[0][0], list):
                return batch_data

        # Fallback: Convert the padded tensor back to a clean list of lists.
        # This removes the trailing 0s (padding_idx) added by the collate_fn/preprocessor.
        raw_seqs = []
        cpu_batch = tensor_batch.cpu().numpy()

        for i in range(len(cpu_batch)):
            board = []
            for r in range(len(cpu_batch[i])):
                row_data = cpu_batch[i][r]
                clean_row = [int(x) for x in row_data if x != self.padding_idx]
                board.append(clean_row)
            raw_seqs.append(board)

        return raw_seqs

    def train_step(self, batch_data):
        """
        Executes a single training step for a batch of data.

        Process:
            1. Forward Pass (Actor) -> Get Gap Distributions.
            2. Sampling -> Get Gap Counts (Actions).
            3. Environment Evaluation -> Get Rewards & Metrics.
            4. Backward Pass -> Update Weights using GRPO Loss.
        """
        # 1. Prepare Data
        # State: (Batch, Rows, Len) - Tensor on Device
        state = self._prepare_batch(batch_data)

        # Raw Sequences: List of Lists (CPU) - The biological ground truth
        raw_sequences = self._get_raw_sequences(batch_data, state)
        batch_size = state.shape[0]

        # 2. Replicate Inputs for Group Generation
        # We expand the batch [A, B] -> [A...A, B...B] (Group Size times)
        # Dimensions: (Batch * Group, Rows, Len)
        state_repeated = state.repeat_interleave(self.group_size, dim=0)

        # 3. Parallel Sampling (Forward Pass)
        # The actor handles distribution creation and masking internally.
        # dist is usually a Normal distribution (Mean, Std).
        dist = self.agent.actor.get_distribution(state_repeated)

        # Sample continuous actions from the distribution
        actions = dist.sample()  # Shape: (Batch*G, Rows, Len)

        # 4. Compute Log Probs (Essential for Gradient Calculation)
        log_probs = dist.log_prob(actions)

        # Mask Log Probs: Padding positions shouldn't affect the gradient.
        # mask is 1 where data is valid, 0 where padding.
        mask = (state_repeated != self.padding_idx).float()

        # Sum log_probs over Rows and Length to get one scalar probability per sample
        # (Assuming independence between positions)
        log_probs = (log_probs * mask).sum(dim=[1, 2])

        # 5. Evaluation (The Bridge to Environment)
        rewards = []

        # We accumulate biological metrics for logging (SP, CS, EM, AL)
        metrics_history = {"SP": [], "CS": [], "EM": [], "AL": []}

        actions_cpu = actions.cpu()

        # A. Decode: Neural Net Output -> Integer Gap Matrix
        # The adapter transforms floats (e.g., 1.7) into integers (e.g., 2).
        # gap_matrix_batch is a List[List[List[int]]]
        gap_matrix_batch = self.agent.actor.adapter.decode(
            actions_cpu  # decode now only needs the tensor
        )

        # B. Evaluate Loop (CPU Bottleneck - iterates over generated samples)
        for i in range(batch_size * self.group_size):
            # Calculate which original sequence this sample belongs to
            original_idx = i // self.group_size

            # Retrieve specific problem instance (Raw DNA)
            current_raw_seqs = raw_sequences[original_idx]

            # Retrieve the specific action (Gap Matrix)
            current_gap_matrix = gap_matrix_batch[i]

            # Instantiate Stateless Environment with the raw DNA
            env = Environment(current_raw_seqs, mode=self.env_mode)

            # === CORE CALL ===
            # The Env reconstructs the alignment by merging DNA + Gap Matrix
            # and calculates the scores.
            score, aligned_seqs_debug, metrics = env.evaluate(current_gap_matrix)
            # =================

            rewards.append(score)

            # Store metrics for averaging later
            for k, v in metrics.items():
                metrics_history[k].append(v)

            # Debug Logging (Periodic check of the first sample)
            if i == 0 and self.global_step % 20 == 0:
                self.logger.info(f"\n--- DEBUG STEP {self.global_step} ---")
                self.logger.info(f"Reward: {score:.2f} | CS: {metrics['CS']:.2f}% | SP: {metrics['SP']:.2f}")

                # Visual check of the first 20 characters of the first row
                debug_map = {1: 'A', 2: 'C', 3: 'G', 4: 'T', 5: '-', 0: 'P'}
                # Robustly map chars, handling potential unknown tokens
                row0 = "".join([debug_map.get(x, '?') for x in aligned_seqs_debug[0][:30]])
                self.logger.info(f"Row 0: {row0}...")
                self.logger.info("------------------------\n")

        # Convert rewards to tensor for PyTorch operations
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)

        # 6. GRPO Advantage Calculation (Z-Score Normalization)
        # Reshape to (Batch, Group) to compare samples against their own group
        rewards_grouped = rewards.view(batch_size, self.group_size)

        # Calculate Group Mean and Std
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8

        # Advantage = (Reward - GroupMean) / GroupStd
        # This encourages the model to prefer actions that are better than the average of its own attempts.
        advantages = (rewards_grouped - group_mean) / group_std
        advantages = advantages.view(-1).detach()  # Flatten back

        # 7. Optimization
        # GRPO Loss = - (Advantage * Log_Prob)
        loss = - (advantages * log_probs).mean()

        self.agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), 1.0)
        self.agent.optimizer.step()

        # 8. Logging to TensorBoard
        # We average the metrics over the batch
        avg_metrics = {k: np.mean(v) for k, v in metrics_history.items()}

        if self.writer:
            self.writer.add_scalar("Train/Loss", loss.item(), self.global_step)
            self.writer.add_scalar("Train/Reward_Raw", rewards.mean().item(), self.global_step)
            self.writer.add_scalar("Metrics/SP_Score", avg_metrics['SP'], self.global_step)
            self.writer.add_scalar("Metrics/CS_Percent", avg_metrics['CS'], self.global_step)
            self.writer.add_scalar("Metrics/Exact_Matches", avg_metrics['EM'], self.global_step)
            self.writer.add_scalar("Metrics/Align_Len", avg_metrics['AL'], self.global_step)

        return {
            'loss': loss.item(),
            'avg_reward': rewards.mean().item(),
            'max_reward': rewards.max().item(),
            'avg_cs': avg_metrics['CS']
        }