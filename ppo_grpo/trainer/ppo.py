import torch
import torch.nn as nn
from .base import BaseTrainer
from ppo_grpo.env import Environment


class PPOTrainer(BaseTrainer):
    """
    Standard PPO Trainer (One-Shot version).
    - Input: Batch of B sub-boards.
    - Process: 
      1. Sample Actions & Values (Old Policy).
      2. Compute Rewards.
      3. Loop K epochs: Update Policy to maximize Reward using Clipped Objective.
    """

    def __init__(self, agent, clip_eps=0.2, entropy_coef=0.01, ppo_epochs=4, env_mode='sp', **kwargs):
        super().__init__(agent, **kwargs)
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.env_mode = env_mode
        self.mse_loss = nn.MSELoss()

    def _get_raw_sequences(self, batch_data, tensor_batch):
        # Same helper as GRPO (could be moved to BaseTrainer if reused often)
        if isinstance(batch_data, list) and not torch.is_tensor(batch_data):
            if hasattr(batch_data[0], 'sequences'):
                return [x.sequences for x in batch_data]
            return batch_data

        raw_seqs = []
        cpu_batch = tensor_batch.cpu().numpy()
        for i in range(len(cpu_batch)):
            sub_board = []
            for row in cpu_batch[i]:
                clean_row = [x for x in row if x != self.padding_idx]
                sub_board.append(clean_row)
            raw_seqs.append(sub_board)
        return raw_seqs

    def train_step(self, batch_data):
        # 1. Prepare Data
        state = self._prepare_batch(batch_data)
        raw_sequences = self._get_raw_sequences(batch_data, state)
        mask = (state != self.padding_idx).float()

        # 2. Rollout Phase (No Gradient)
        # Collect "Old" Trajectory data
        with torch.no_grad():
            old_dist = self.agent.actor.get_distribution(state)
            old_action = old_dist.sample()

            # Compute Old Log Probs (masked)
            old_log_prob = old_dist.log_prob(old_action)
            old_log_prob = (old_log_prob * mask).sum(dim=[1, 2])

            # Compute Old Value
            # Critic output shape (B, 1) -> Squeeze to (B)
            old_value = self.agent.critic.get_value(state).squeeze(-1)

            # Compute Rewards (Environment)
            rewards = []
            old_action_cpu = old_action.cpu()
            for i in range(len(raw_sequences)):
                aligned_seqs = self.agent.actor.adapter.decode(
                    raw_sequences[i],
                    old_action_cpu[i]
                )
                env = Environment(raw_sequences[i], mode=self.env_mode)
                rewards.append(env.compute_reward(aligned_seqs))

            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)

            # Calculate Advantage
            # For One-Shot: Returns = Reward. 
            # Advantage = Reward - Baseline (Value)
            advantages = rewards - old_value

            # Normalize advantages (Standard PPO trick for stability)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. PPO Update Phase (With Gradient)
        # We re-process the SAME batch multiple times
        total_p_loss = 0
        total_v_loss = 0

        for _ in range(self.ppo_epochs):
            # Evaluate current policy on old actions
            new_dist = self.agent.actor.get_distribution(state)
            new_log_prob = new_dist.log_prob(old_action)
            new_log_prob = (new_log_prob * mask).sum(dim=[1, 2])

            entropy = new_dist.entropy()
            entropy_loss = (entropy * mask).sum(dim=[1, 2]).mean()

            new_value = self.agent.critic.get_value(state).squeeze(-1)

            # Ratio calculation
            ratio = torch.exp(new_log_prob - old_log_prob)

            # Surrogate Losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages

            # PPO Loss (Maximize objective -> Minimize negative)
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value Loss (MSE)
            value_loss = self.mse_loss(new_value, rewards)

            # Total Loss
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy_loss

            self.agent.optimizer.zero_grad()
            loss.backward()

            # Gather all optimizer groups
            all_params = []
            for group in self.agent.optimizer.param_groups:
                all_params.extend(group['params'])
            # Clipping all groups params
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

            self.agent.optimizer.step()

            total_p_loss += policy_loss.item()
            total_v_loss += value_loss.item()

        return {
            'loss': (total_p_loss + total_v_loss) / self.ppo_epochs,
            'policy_loss': total_p_loss / self.ppo_epochs,
            'value_loss': total_v_loss / self.ppo_epochs,
            'avg_reward': rewards.mean().item()
        }