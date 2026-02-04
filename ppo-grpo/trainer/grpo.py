import torch
from .base import BaseTrainer
from ..env import Environment


class GRPOTrainer(BaseTrainer):
    """
    Generative RPO Trainer.
    - Input: Batch of B sub-boards.
    - Process: Replicates inputs G times -> B*G samples.
    - Update: Uses Group Relative Policy Optimization (Action - GroupMean).
    """

    def __init__(self, agent, group_size=8, grpo_epochs=4, clip_eps=0.2, env_mode='sp', **kwargs):
        super().__init__(agent, **kwargs)
        self.group_size = group_size
        self.grpo_epochs = grpo_epochs
        self.clip_eps = clip_eps
        self.env_mode = env_mode

    def _get_raw_sequences(self, batch_data, tensor_batch):
        """
        Helper to get clean list-of-lists (without padding) for the Environment.
        """
        # If the dataloader already gave us lists, use them (faster)
        if isinstance(batch_data, list) and not torch.is_tensor(batch_data):
            if hasattr(batch_data[0], 'sequences'):
                return [x.sequences for x in batch_data]
            return batch_data

        # If input was pure Tensor (HDF5), we must convert back to list and strip padding
        raw_seqs = []
        cpu_batch = tensor_batch.cpu().numpy()
        for i in range(len(cpu_batch)):
            sub_board = []
            for row in cpu_batch[i]:
                # Filter out padding_idx
                clean_row = [x for x in row if x != self.padding_idx]
                sub_board.append(clean_row)
            raw_seqs.append(sub_board)
        return raw_seqs

    def train_step(self, batch_data):
        # 1. Prepare Data
        # State: (Batch, Rows, Len)
        state = self._prepare_batch(batch_data)
        raw_sequences = self._get_raw_sequences(batch_data, state)
        batch_size = state.shape[0]

        # 2. Replicate Inputs for Group Generation
        # [A, B] -> [A...A, B...B] (Group Size times)
        state_repeated = state.repeat_interleave(self.group_size, dim=0)

        # 3. Parallel Sampling (Forward Pass)
        # The actor handles distribution creation and masking internally
        dist = self.agent.actor.get_distribution(state_repeated)
        actions = dist.sample()  # (Batch*G, Rows, Len)

        # 4. Compute Log Probs
        log_probs = dist.log_prob(actions)

        # Mask Log Probs (Padding shouldn't affect gradient)
        # mask is 1 where data is valid, 0 where padding
        mask = (state_repeated != self.padding_idx).float()
        # Sum over Rows and Length to get one probability per sample
        log_probs = (log_probs * mask).sum(dim=[1, 2])

        # 5. Evaluation (Oracle/Env) - CPU Bottleneck
        rewards = []
        actions_cpu = actions.cpu()

        for i in range(batch_size * self.group_size):
            # Map repeated index back to original batch index
            original_idx = i // self.group_size

            # Use Adapter to Decode (Numbers -> Gaps)
            # We pass the raw sequence so it knows the length
            aligned_seqs = self.agent.actor.adapter.decode(
                raw_sequences[original_idx],
                actions_cpu[i]
            )

            # Compute Score
            # We create a temp env for scoring.
            # Note: Environment should be the stateless version.
            env = Environment(raw_sequences[original_idx], mode=self.env_mode)
            rew = env.compute_reward(aligned_seqs)
            rewards.append(rew)

        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)

        # 6. GRPO Advantage Calculation
        # Reshape to (Batch, Group)
        rewards_grouped = rewards.view(batch_size, self.group_size)

        # Calculate Group Mean and Std
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8

        # Advantage = Z-Score within the group
        advantages = (rewards_grouped - group_mean) / group_std
        advantages = advantages.view(-1).detach()  # Flatten

        # 7. Optimization
        # GRPO Loss = - (Advantage * Log_Prob)
        loss = - (advantages * log_probs).mean()

        self.agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), 1.0)
        self.agent.optimizer.step()

        return {
            'loss': loss.item(),
            'avg_reward': rewards.mean().item(),
            'max_reward': rewards.max().item()
        }