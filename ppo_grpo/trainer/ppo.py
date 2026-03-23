import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseTrainer
from ppo_grpo.agent.ppo_agent import PPO_Agent

# New Import: The Vectorized GPU Environment
from ppo_grpo.vect_env import VectorizedEnvironment


class PPOTrainer(BaseTrainer[PPO_Agent]):
    """
    Standard PPO Trainer (One-Shot version) - GPU Accelerated.

    This trainer implements Proximal Policy Optimization (PPO) for Multiple Sequence Alignment.
    It replaces the CPU-based alignment simulation with a fully vectorized GPU approach.

    Process:
      1. Experience Collection:
         - Sample actions (Gap Counts) from the Current Policy (Old Policy).
         - Compute Rewards efficiently on GPU using `VectorizedGPUEnvironment`.
         - Compute Advantages using the Critic's value estimate (A = Reward - Value).
      2. Optimization Loop (PPO Epochs):
         - Recalculate probabilities for the sampled actions under the updating policy.
         - Compute the PPO Clipped Surrogate Objective.
         - Update both Actor and Critic networks.
    """

    def __init__(
            self,
            agent,
            clip_eps=0.2,
            ppo_epochs=4,
            env_mode='sp',
            value_loss_coef=0.5,  # Added explicit value loss coefficient
            **kwargs
    ):
        """
        Args:
            agent: The PPO Agent containing Actor and Critic networks.
            clip_eps: PPO clipping parameter (epsilon), typically 0.1 or 0.2.
            ppo_epochs: Number of times to update the policy on the same batch.
            env_mode: 'sp' (Sum-of-Pairs) or 'cs' (Column Score) - handled by Config now.
            entropy_coef: Coefficient for entropy regularization.
            value_loss_coef: Coefficient for the Value function loss.
        """
        super().__init__(agent, **kwargs)
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.env_mode = env_mode
        self.value_loss_coef = value_loss_coef

        # Loss function for the Critic (Value Head)
        self.mse_loss = nn.MSELoss()

        self.env = VectorizedEnvironment(self.config, self.device)

    def train_step(self, batch_data) -> dict[str, float]:
        """
        Executes a single optimized PPO training step.

        Pipeline:
            1. Data Prep: Load batch to GPU.
            2. Rollout:
               - Sample actions (Old Policy).
               - Score alignments (GPU Vectorized).
               - Calculate Advantages (Reward - Value).
            3. PPO Update:
               - Loop `ppo_epochs` times.
               - Calculate Ratio (New/Old).
               - Compute Clipped Loss.
               - Update Actor and Critic.

        Args:
            batch_data: Raw batch from DataLoader.

        Returns:
            metrics: Dictionary containing Loss, Rewards, and Biological Metrics.
        """

        # PHASE 1: EXPERIENCE COLLECTION (ROLLOUT)

        # 1. Prepare Data
        # state shape: (Batch, Rows, Len) -> On GPU
        state = self._prepare_batch(batch_data)

        # Create mask for valid positions (1.0 for DNA, 0.0 for Padding)
        mask = self.agent.build_mask(state).float()

        # 2. Rollout (No Gradients)
        # We collect the trajectory defined by the "Old" policy.
        with torch.no_grad():
            # A. Get Old Policy Distribution
            old_dist = self.agent.actor.get_distribution(state, mask)

            # B. Sample Actions (Continuous)
            old_action_float = old_dist.sample()  # Shape: (Batch, Rows, Len)

            # C. Compute Old Log Probabilities
            # Sum over dimensions (Rows, Len) to get probability per sample.
            old_log_prob_raw = old_dist.log_prob(old_action_float)
            old_log_prob = (old_log_prob_raw * mask).sum(dim=[1, 2])

            # D. Get Old Value Estimate (Critic)
            # Critic output shape (Batch, 1) -> Squeeze to (Batch)
            old_value = self.agent.critic.get_value(state).squeeze(-1)

            # E. Decode & Evaluate (GPU Environment) 🚀
            # Convert continuous actions to integers
            # Rounding breaks the graph, which is correct for interaction with Env.
            actions_int = torch.round(F.relu(old_action_float)).long()

            # Run the vectorized environment (Massively Parallel)
            rewards, metrics = self.env.evaluate_batch(state, actions_int)

            # F. Calculate Advantages & Returns
            # In a "One-Shot" bandit setting like this (Episode Length = 1 step):
            # Return = Reward (No discount factor gamma needed)
            returns = rewards

            # Advantage = Actual Reward - Estimated Value
            advantages = returns - old_value

            # G. Normalize Advantages (Standard PPO trick for stability)
            # Shifts mean to 0 and std to 1.
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PHASE 2: OPTIMIZATION LOOP (PPO UPDATE)

        total_p_loss = 0.0
        total_v_loss = 0.0
        total_ent = 0.0
        final_loss = 0.0

        # We re-process the SAME batch multiple times to squeeze more learning signal.
        for _ in range(self.ppo_epochs):
            # A. Evaluate Current Policy
            # We get the distribution again because the weights might have updated
            # in the previous iteration of this loop.
            new_dist = self.agent.actor.get_distribution(state)

            # B. New Log Probs
            # Probability of the actions sampled in Phase 1, but under the NEW policy.
            new_log_prob_raw = new_dist.log_prob(old_action_float)
            new_log_prob = (new_log_prob_raw * mask).sum(dim=[1, 2])

            # C. New Value Estimate
            new_value = self.agent.critic.get_value(state).squeeze(-1)

            # D. Ratio Calculation (Importance Sampling)
            # Ratio = exp(log_new - log_old)
            ratio = torch.exp(new_log_prob - old_log_prob)

            # E. PPO Clipped Surrogate Objective
            # 1. Unclipped: Ratio * Advantage
            surr1 = ratio * advantages

            # 2. Clipped: Clamp(Ratio) * Advantage
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages

            # 3. Policy Loss: Minimize negative of the lower bound
            policy_loss = -torch.min(surr1, surr2).mean()

            # F. Value Loss
            # MSE between the Critic's prediction and the actual observed Reward.
            # (Sometimes value clipping is used here too, but simple MSE is standard).
            value_loss = self.mse_loss(new_value, returns)

            # G. Entropy Regularization
            # Encourage exploration
            weighted_entropy, mean_entropy = self._compute_masked_entropy(new_dist, mask)

            # H. Total Loss
            # Loss = Policy Loss + c1 * Value Loss - c2 * Entropy
            # Note: We subtract entropy because we want to MAXIMIZE it (minimize negative).
            loss = policy_loss + (self.value_loss_coef * value_loss) - weighted_entropy

            # I. Backpropagation
            self.agent.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            # We clip all parameters managed by the optimizer
            all_params = []
            for group in self.agent.optimizer.param_groups:
                all_params.extend(group['params'])
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

            self.agent.optimizer.step()

            # J. Accumulate Metrics
            total_p_loss += policy_loss.item()
            total_v_loss += value_loss.item()
            total_ent += mean_entropy.item()
            final_loss = loss.item()

        # 3. Metrics Return
        return {
            'loss': final_loss,
            'loss_policy': total_p_loss / self.ppo_epochs,
            'loss_value': total_v_loss / self.ppo_epochs,
            'entropy': total_ent / self.ppo_epochs,
            'avg_reward': rewards.mean().item(),
            'max_reward': rewards.max().item(),
            'SP_Score': metrics['SP'],  # From the rollout evaluation
            'CS_Percent': metrics['CS'],
            'Alignment_Len': metrics['AL']
        }