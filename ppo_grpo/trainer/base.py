import torch
import os
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
from logging import Logger
from typing import TypeVar, Generic

from ppo_grpo.data import IMSAPreprocessor
from ppo_grpo.agent import BaseMSAAgent

AgentType = TypeVar("AgentType", bound=BaseMSAAgent)

class BaseTrainer(ABC, Generic[AgentType]):
    """
    Abstract base class for MSA Trainers.
    Handles common boilerplate:
    - Device management
    - Checkpointing
    - Tensorboard logging
    - Batch preparation (Dynamic Padding)
    """

    def __init__(
            self,
            agent : AgentType,
            preprocessor: IMSAPreprocessor,
            config,
            logger: Logger,
            writer: SummaryWriter | None = None,
            output_dir: str = "checkpoints",
            padding_idx: int = 5,
            entropy_coef: float = 0.0,
    ):
        """
        Args:
            agent: The PPO_Agent or GRPO_Agent instance.
            config: A configuration object/module containing hyperparameters.
            logger: The python logger for console output.
            writer: TensorBoard SummaryWriter (optional).
            output_dir: Path to save model weights.
            padding_idx: The integer value used to pad shorter sequences in a batch.
        """
        self.agent = agent
        self.preprocessor = preprocessor
        self.config = config
        self.logger = logger
        self.writer = writer
        self.output_dir = output_dir
        self.device = agent.device
        self.padding_idx = padding_idx  # Store it

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        self.global_step = 0
        config_entropy_coef = getattr(config, "ENTROPY_COEFFICIENT", None)
        self.entropy_coef = config_entropy_coef if config_entropy_coef is not None else entropy_coef

        # DEBUG
        logger.info(f"Entropy Coefficient: {self.entropy_coef}")

    def save_checkpoint(self, filename: str):
        filepath = os.path.join(self.output_dir, filename)
        self.agent.save(filepath)
        self.logger.info(f"- Checkpoint saved: {filepath}")

    def log_metrics(self, metrics: dict[str, float], step: int, prefix: str = "Train"):
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"{prefix}/{key}", value, step)

    def _prepare_batch(self, batch_data) -> torch.Tensor:
        """
        Prepares the batch for the network.
        Now it simply delegates the work to the preprocessor.
        """
        return self.preprocessor(batch_data, sanitize=True)

    def _compute_masked_entropy(
            self,
            dist: torch.distributions.Distribution,
            mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the mean entropy of the distribution, ignoring padding.

        Standard PyTorch .mean() would divide by the total size of the tensor
        (including padding), artificially lowering entropy for short sequences.
        This method computes the mean entropy per VALID token.

        Formula:
            E = Sum(H(x) * mask) / Sum(mask)

        Args:
            dist: The PyTorch distribution output by the Actor.
                  Shape: (Batch, Rows, Len)
            mask: Boolean or Float tensor indicating valid data (1) vs padding (0).
                  Shape: (Batch, Rows, Len)

        Returns:
            weighted_entropy (torch.Tensor): mean_entropy * entropy_coef
            raw_mean_entropy (torch.Tensor): A scalar tensor representing the average uncertainty
                          per valid nucleotide.
        """
        # 1. Get raw entropy for every position (Batch, Rows, Len)
        raw_entropy = dist.entropy()

        if raw_entropy.shape != mask.shape:
            self.logger.warning(f"Entropy shape {raw_entropy.shape} != Mask shape {mask.shape}")

        # 2. Zero out entropy values corresponding to padding tokens.
        masked_entropy = raw_entropy * mask

        # 3. Compute the mean over valid tokens only.
        valid_token_count = mask.sum() + 1e-8

        mean_entropy = masked_entropy.sum() / valid_token_count

        #4 Compute weighted term
        weighted_entropy = mean_entropy * self.entropy_coef

        return weighted_entropy, mean_entropy

    @abstractmethod
    def train_step(self, batch) -> dict[str, float]:
        pass

    def train_epoch(self, dataloader, epoch_idx: int):
        self.agent.train()

        for batch_idx, batch_data in enumerate(dataloader):
            metrics = self.train_step(batch_data)
            self.global_step += 1

            if batch_idx % 10 == 0:
                self.log_metrics(metrics, self.global_step)

            if batch_idx % 100 == 0:
                loss_str = f"{metrics.get('loss', 0):.4f}"
                rew_str = f"{metrics.get('avg_reward', 0):.2f}"
                self.logger.info(
                    f"Epoch {epoch_idx} | Step {batch_idx} | "
                    f"Loss: {loss_str} | Reward: {rew_str}"
                )