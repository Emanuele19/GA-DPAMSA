import torch
import torch.nn.functional as F
from ..interfaces import IMSAOutputAdapter
import config


class GaussianGapAdapter(IMSAOutputAdapter):
    """
    An adapter that models gap insertion as a continuous Gaussian distribution.

    Instead of predicting probabilities for every possible gap count (0, 1, 2...),
    the model predicts a Mean and Standard Deviation. The sampled value is then 
    rectified (ReLU) and rounded to the nearest integer.

    Output Dimension: 2 per slot (Mean, Log_Std).
    """

    def __init__(
            self,
            max_gaps: int = getattr(config, "MAX_GAPS", None),
            min_log_std: float = getattr(config, "MIN_LOG_STD", -20.0),
            max_log_std: float = getattr(config, "MAX_LOG_STD", 2.0),
    ):
        self.max_gaps = max_gaps

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    @property
    def logit_dim(self) -> int:
        """
        Returns 2: One for Mean, one for Log Standard Deviation.
        """
        return 2

    def logits_to_dist(
            self,
            logits: torch.Tensor,
            attention_mask: torch.Tensor | None = None
    ) -> torch.distributions.Distribution:
        """
        Converts logits into a Normal distribution.

        Args:
            logits: (Batch, Rows, Len, 2). 
                    logits[..., 0] is the Mean.
                    logits[..., 1] is the Log_Std (to ensure Std is positive).
            attention_mask: (Batch, Rows, Len).
                    defines the output mask.

        """

        # Using softplus to avoid dead neurons
        mu = F.softplus(logits[..., 0])
        log_std = logits[..., 1]

        # --- MASKING LOGIC ---
        if attention_mask is not None:
            # attention_mask is True for valid data, False for padding.
            # We want to force padding positions to have:
            # Mean = 0, Log_Std = -5 (Small variance)

            # Cast mask to float (1.0 for valid, 0.0 for padding)
            mask_float = attention_mask.float()

            # 1. Zero out the Mean for padding
            mu = mu * mask_float

            # 2. Log_Std for padding
            # Std for padding is set an infinitesimal value
            # This avoids any useless learning made outside the mask
            log_std = log_std * mask_float + (1.0 - mask_float) * self.min_log_std

        # Clamping for stability (as before)
        log_std = torch.clamp(log_std, min=self.min_log_std, max=self.max_log_std)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mu, std)
        return dist

    def decode(self, actions: torch.Tensor) -> list[list[list[int]]]:
        """
        Transforms sampled continuous values into valid gap integer lists.

        Clean Architecture Version:
        - It acts on the full tensor.
        - It doesn't care about biological sequence lengths.
        - Trimming is left to the consumer (Environment).
        """
        # 1. Apply constraints: ReLU (non-negative) -> Round (integer)
        cleaned_actions = torch.round(F.relu(actions)).int()

        if self.max_gaps:
            cleaned_actions = torch.clamp(cleaned_actions, max=self.max_gaps)

        # 2. Convert to Python List of Lists
        if hasattr(cleaned_actions, 'cpu'):
            cleaned_actions = cleaned_actions.cpu()

        # Returns a list of lists, e.g., 30 integers per row
        return cleaned_actions.tolist()