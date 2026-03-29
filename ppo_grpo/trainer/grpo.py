import torch
import torch.nn.functional as F
from .base import BaseTrainer
from ppo_grpo.vect_env import VectorizedEnvironment
from ppo_grpo.agent.grpo_agent import GRPO_Agent



class GRPOTrainer(BaseTrainer[GRPO_Agent]):
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

    def __init__(
            self,
            agent,
            group_size=8,
            grpo_epochs=4,
            clip_eps=0.2,
            env_mode='sp',
            **kwargs
    ):
        super().__init__(agent, **kwargs)
        self.group_size = group_size
        self.grpo_epochs = grpo_epochs
        self.clip_eps = clip_eps
        self.env_mode = env_mode

        self.env = VectorizedEnvironment(self.config, self.device)

    def train_step(self, batch_data) -> dict[str, float]:
        """
        Executes a single optimization step using GRPO (Group Relative Policy Optimization)
        with PPO Clipping.

        This method performs two distinct phases:
        1. **Experience Collection (Sampling Phase):**
           - Generates multiple variations (Group) for each input sequence.
           - Evaluates them on the GPU Environment to get Rewards.
           - Calculates Advantages relative to the group mean.

        2. **Optimization (Training Phase):**
           - Iterates `grpo_epochs` times over the collected experience.
           - Recalculates probabilities (New Policy) vs original probabilities (Old Policy).
           - Applies PPO Clipping to prevent destructive updates.

        Args:
            batch_data: Raw batch from the DataLoader.

        Returns:
            metrics: A dictionary of scalar metrics (Loss, Reward, Entropy, etc.) for logging.
        """

        # =====================================================================
        # PHASE 1: EXPERIENCE COLLECTION (SAMPLING)
        # We generate data using the CURRENT policy (which becomes 'OLD' for the PPO loop).
        # =====================================================================

        # 1. Prepare Data
        # state shape: (Batch, Rows, Len) -> On GPU
        state = self._prepare_batch(batch_data)
        batch_size = state.shape[0]

        # 2. Expansion (Group Generation)
        # Replicate input sequences `group_size` times.
        # shape: (Batch * Group, Rows, Len)
        state_repeated = state.repeat_interleave(self.group_size, dim=0)

        mask = self.agent.build_mask(state_repeated).float()

        # 3. Action Sampling & Evaluation
        # We use no_grad because we are collecting fixed data points (trajectory)
        # The gradients will be calculated later during the PPO optimization loop.
        with torch.no_grad():
            # A. Get Old Policy Distribution
            old_dist = self.agent.actor.get_distribution(state_repeated, mask=mask)

            # B. Sample Actions (Continuous)
            actions_float = old_dist.sample()  # Shape: (Batch*Group, Rows, Len)

            # C. Compute Old Log Probabilities
            # These are the probabilities of the actions *at the time of sampling*.
            # They serve as the denominator in the PPO Ratio (New/Old).
            # Sum log_probs over dimensions (Rows, Len) to get probability per sample.
            old_log_probs_raw = old_dist.log_prob(actions_float)
            old_log_probs = (old_log_probs_raw * mask).sum(dim=[1, 2])

            # D. Decode & Evaluate (GPU Environment)
            # Convert continuous actions to integers for the environment
            actions_int = torch.round(F.softplus(actions_float)).long()

            # Run the vectorized environment (Massively Parallel)
            rewards, metrics = self.env.evaluate_batch(state_repeated, actions_int)

        # 4. GRPO Advantage Calculation
        # We normalize rewards within each group to reduce variance.
        # Shape: (Batch, Group)
        rewards_grouped = rewards.view(batch_size, self.group_size)

        # Calculate Group Stats
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8

        # raw_group_std = rewards_grouped.std(dim=1, keepdim=True)
        #
        # # Log the average standard deviation within groups.
        # # If this is 0, the model has collapsed (all samples are identical).
        # debug_avg_group_std = raw_group_std.mean().item()

        # Advantage = Z-Score of the reward within its own group
        advantages = (rewards_grouped - group_mean) / group_std

        # Flatten back to match the flat batch structure: (Batch * Group)
        advantages = advantages.view(-1)

        # =====================================================================
        # PHASE 2: OPTIMIZATION LOOP (PPO UPDATE)
        # We update the policy multiple times using the collected experience.
        # =====================================================================

        total_loss = 0.0
        total_entropy = 0.0
        #total_grad_norm = 0.0

        for _ in range(self.grpo_epochs):
            # A. Forward Pass (New Policy)
            # The network weights have ostensibly changed (or will change),
            # so we get the 'New' distribution for the SAME state.
            new_dist = self.agent.actor.get_distribution(state_repeated, mask=mask)

            # B. New Log Probs
            # We evaluate the probability of the *originally sampled actions* # under the *current/new* policy.
            new_log_probs_raw = new_dist.log_prob(actions_float)
            new_log_probs = (new_log_probs_raw * mask).sum(dim=[1, 2])

            # C. Ratio Calculation (Importance Sampling)
            # Ratio = P_new / P_old
            # Computed in log space for numerical stability: exp(log_new - log_old)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # D. PPO Clipping Logic (Surrogate Objective)
            # 1. Unclipped Objective: Ratio * Advantage
            surr1 = ratio * advantages

            # 2. Clipped Objective: Clamp Ratio to [1-eps, 1+eps] * Advantage
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages

            # 3. Maximize the lower bound (conservative update)
            # Since we minimize loss, we take negative of the minimum.
            ppo_loss = -torch.min(surr1, surr2).mean()

            # E. Entropy Regularization
            # Encourage exploration by maximizing entropy of the NEW distribution
            weighted_entropy, mean_entropy = self._compute_masked_entropy(new_dist, mask)

            # F. Total Loss
            loss = ppo_loss - weighted_entropy

            # G. Backpropagation
            self.agent.optimizer.zero_grad()
            loss.backward()

            # # --- DEBUG: CHECK GRADIENT NORMS ---
            # # Calculates the L2 norm of gradients to ensure the network is learning.
            # current_norm = 0.0
            # for p in self.agent.actor.parameters():
            #     if p.grad is not None:
            #         param_norm = p.grad.data.norm(2)
            #         current_norm += param_norm.item() ** 2
            # current_norm = current_norm ** 0.5
            # total_grad_norm += current_norm

            # Gradient Clipping (Prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), 1.0)

            self.agent.optimizer.step()

            # Accumulate stats for averaging
            total_loss += loss.item()
            total_entropy += mean_entropy.item()

        # =====================================================================
        # 3. METRICS AGGREGATION
        # =====================================================================

        # Average metrics over the PPO epochs
        avg_loss = total_loss / self.grpo_epochs
        avg_entropy = total_entropy / self.grpo_epochs

        # # Statistics about the actions (Are we predicting gaps?)
        # avg_gaps_pred = actions_int.float().mean().item()
        # max_gaps_pred = actions_int.max().item()

        return {
            'loss': avg_loss,
            'entropy': avg_entropy,
            'avg_reward': rewards.mean().item(),
            'min_reward': rewards.min().item(),
            'max_reward': rewards.max().item(),
            'SP_Score': metrics['SP'],  # From the last evaluation pass
            'CS_Percent': metrics['CS'],
            'Alignment_Len': metrics['AL'],

            # # --- CRITICAL DEBUG METRICS ---
            # # If Group_Std is 0 or very low (<1e-5), increase Adapter Sigma!
            # 'Debug/Group_Std': debug_avg_group_std,
            #
            # # If Grad_Norm is 0, the backward pass is broken.
            # 'Debug/Grad_Norm': total_grad_norm / self.grpo_epochs,
            #
            # # If Avg_Gaps is 0, the model is lazy (increase Entropy coef).
            # 'Debug/Avg_Gaps': avg_gaps_pred,
            # 'Debug/Max_Gaps': float(max_gaps_pred),
        }