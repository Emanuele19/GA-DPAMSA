import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class IMSAOutputAdapter(ABC):
    """
    Interface for handling the translation between Neural Network output (logits)
    and Domain Action (Gap Insertion).

    Responsibilities:
    1. Define the shape of the output layer.
    2. Convert raw logits into a PyTorch Distribution (for PPO/GRPO sampling).
    3. Decode sampled actions back into aligned biological sequences.
    """

    @property
    @abstractmethod
    def logit_dim(self) -> int:
        """
        Returns the number of output classes per decision slot.
        Example: If we allow 0 to 5 gaps, this returns 6.
        """
        pass

    @abstractmethod
    def logits_to_dist(self,
                       logits: torch.Tensor,
                       attention_mask: torch.Tensor | None = None) -> torch.distributions.Distribution:
        """
        Converts raw logits into a sampleable distribution.

        Args:
            logits: Tensor of shape (Batch, Rows, Len, logit_dim).
            attention_mask: Boolean tensor of shape (Batch, Rows, Len). 
                            True indicates valid data, False indicates padding.
                            Used to mask out probabilities for padding tokens.

        Returns:
            A PyTorch Distribution (e.g., Categorical) ready for sampling.
        """
        pass

    @abstractmethod
    def decode(self,
               raw_sequences: list[list[int]],
               actions: torch.Tensor) -> list[list[int]]:
        """
        Reconstructs the final alignment based on the actions.

        Args:
            raw_sequences: The original input sequences (integers).
            actions: Tensor of shape (Batch, Rows, Len) containing integer decisions.

        Returns:
            list of aligned sequences with gaps inserted.
        """
        pass


class IMSABackbone(nn.Module, ABC):
    """
    Interface for the Feature Extractor (Encoder).
    It transforms raw sequence indices into latent feature vectors.
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """
        Returns the size of the hidden feature vector (embedding size).
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            x: Input tensor of shape (Batch, Rows, Max_Len).
            mask: Attention mask (Batch, Rows, Max_Len). 
                  True for valid tokens, False for padding.

        Returns:
            Tensor of shape (Batch, Rows, Max_Len, output_dim).
            It must preserve the spatial structure (Rows, Len).
        """
        pass


class IMSAActor(nn.Module, ABC):
    """
    Interface for the Policy Model (Actor).
    It combines a Backbone and an OutputAdapter to make decisions.
    """

    def __init__(self, backbone: IMSABackbone, adapter: IMSAOutputAdapter):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter

    @abstractmethod
    def get_distribution(self,
                         state: torch.Tensor,
                         mask: torch.Tensor | None = None) -> torch.distributions.Distribution:
        """
        Returns the probability distribution over actions for a given state.
        """
        pass

    @abstractmethod
    def get_action(self,
                   state: torch.Tensor,
                   mask: torch.Tensor | None = None,
                   deterministic: bool = False) -> torch.Tensor:
        """
        Helper method for inference. Returns specific actions (integers).

        Args:
            state: Input tensor.
            mask: Optional attention mask.
            deterministic: If True, takes the mode (argmax). If False, samples.
        """
        pass


class IMSACritic(nn.Module, ABC):
    """
    Interface for the Value Function (Critic).
    Used primarily in PPO to estimate V(s).
    """

    def __init__(self, backbone: IMSABackbone):
        super().__init__()
        self.backbone = backbone

    @abstractmethod
    def get_value(self,
                  state: torch.Tensor,
                  mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Estimates the value of the state.

        Args:
            state: Input tensor.
            mask: Optional attention mask.

        Returns:
            Scalar tensor of shape (Batch, 1).
        """
        pass