import torch
import os
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
from logging import Logger


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
            agent,
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
        Robust batch preparation with parameterized padding.
        """
        # Case 1: Already a Tensor
        if torch.is_tensor(batch_data):
            return batch_data.long().to(self.device)

        # Case 2: List of jagged arrays/objects
        if isinstance(batch_data, list):
            # Handle Friend's MSAAlignment object
            if hasattr(batch_data[0], 'sequences'):
                batch_data = [item.sequences for item in batch_data]

            # Find max dimensions
            max_len = 0
            n_rows = len(batch_data[0])

            for sub_board in batch_data:
                curr_len = len(sub_board[0])
                if curr_len > max_len:
                    max_len = curr_len

            batch_size = len(batch_data)

            # Initialize with the specific padding index
            # Using torch.full ensures we fill the void with 'padding_idx' (e.g. -1 or 0)
            padded = torch.full(
                (batch_size, n_rows, max_len),
                fill_value=self.padding_idx,
                dtype=torch.long
            )

            for i, sub_board in enumerate(batch_data):
                # Convert sub_board to tensor
                sb_tensor = torch.tensor(sub_board, dtype=torch.long)
                cols = sb_tensor.shape[1]

                # Copy data into the padded canvas
                padded[i, :, :cols] = sb_tensor

            return padded.to(self.device)

        raise TypeError(f"Unknown batch data type: {type(batch_data)}")

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