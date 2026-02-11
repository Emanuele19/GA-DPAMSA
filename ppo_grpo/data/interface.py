import torch
from typing import Any
from abc import ABC, abstractmethod

class IMSAPreprocessor(ABC):
    """
    Interface for the 'Middleware' component that sits between raw data sources
    (Dataset, Genetic Algorithm) and the Agent.

    Responsibilities:
    1. Sanitize input (remove existing gaps if needed).
    2. Normalize input format (handle lists vs objects).
    3. Pad sequences to form a rectangular tensor.
    4. Move data to the correct computing device (CPU/GPU).
    """

    @abstractmethod
    def __call__(self, batch_data: list[list[int]] | Any, sanitize: bool = True) -> torch.Tensor:
        """
        Converts a batch of raw data into a processed Tensor ready for the Agent.

        Args:
            batch_data: The input data. Can be a list of lists (integers)
                        or a list of Dataset objects.
            sanitize (bool): If True, specific tokens (like alignment gaps)
                             should be removed to allow the model to predict them
                             from scratch ('Gap Healing').

        Returns:
            torch.Tensor: A tensor of shape (Batch, Rows, Max_Len) on the configured device.
        """
        pass