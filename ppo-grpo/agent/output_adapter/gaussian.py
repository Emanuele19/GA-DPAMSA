import numpy as np
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

    def __init__(self, max_gaps: int = getattr(config, "MAX_GAPS", None)):
        self.max_gaps = max_gaps

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
        mu = logits[..., 0]
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
            log_std = log_std * mask_float + (-20.0) * (1.0 - mask_float)

        # Clamping for stability (as before)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mu, std)
        return dist

    def decode(self,
               raw_sequences: list[list[int]],
               actions: torch.Tensor) -> list[list[int]]:
        """
        Transforms sampled continuous values into valid gap integers.

        Process:
        1. Float Action (e.g., -0.5, 1.3, 4.8)
        2. ReLU -> (0.0, 1.3, 4.8) -> No negative gaps
        3. Round -> (0, 1, 5) -> Nearest integer
        4. Clamp -> (0, 1, MAX) -> Safety limit
        """
        # 1. Apply constraints
        # ReLU ensures we don't try to insert "-1" gaps.
        # Round converts "1.7 gaps" to "2 gaps".
        cleaned_actions = torch.round(F.relu(actions)).int()

        # 2. Convert to list of lists for the Environment
        # We also truncate to the actual sequence length (ignoring padding predictions)
        final_alignment_matrix = []

        # Move to CPU for list processing
        cleaned_actions_cpu = cleaned_actions.cpu().numpy()

        for r, seq in enumerate(raw_sequences):
            seq_len = len(seq)
            # Take only the slots relevant to this sequence length
            # shape: (Max_Slots,) -> slice to (Seq_Len,)
            row_gaps = cleaned_actions_cpu[r, :seq_len]

            # Enforce max cap if configured (optional safety)
            row_gaps = np.clip(row_gaps, 0, self.max_gaps)

            final_alignment_matrix.append(row_gaps.tolist())

        return final_alignment_matrix