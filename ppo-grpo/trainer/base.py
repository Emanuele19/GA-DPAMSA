import torch
import os
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
from logging import Logger

from ..data import IMSAPreprocessor
from ..agent import BaseMSAAgent


class BaseTrainer(ABC):
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
            agent : BaseMSAAgent,
            preprocessor: IMSAPreprocessor,
            config,
            logger: Logger,
            writer: SummaryWriter | None = None,
            output_dir: str = "checkpoints",
            padding_idx: int = 5
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

    def save_checkpoint(self, filename: str):
        filepath = os.path.join(self.output_dir, filename)
        self.agent.save(filepath)
        self.logger.info(f"💾 Checkpoint saved: {filepath}")

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

    @abstractmethod
    def train_step(self, batch) -> dict[str, float]:
        pass

    def train_epoch(self, dataloader, epoch_idx: int):
        self.agent.actor.train()

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