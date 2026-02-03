import torch
import torch.nn as nn

from ..interfaces import  IMSAActor, IMSABackbone, IMSAOutputAdapter

class MSA_Actor(IMSAActor):
    """
    Concrete implementation of the Actor (Policy).
    It wraps a specific Backbone (Encoder) and a specific Output Adapter.
    """

    def __init__(self, backbone: IMSABackbone, adapter: IMSAOutputAdapter):
        super().__init__(backbone, adapter)

        # The Backbone gives us features of size 'hidden_dim'.
        # The Adapter tells us it needs 'logit_dim' numbers per position
        # (e.g., 2 for Gaussian, 6 for Categorical).
        self.head = nn.Linear(backbone.output_dim, adapter.logit_dim)

    def get_distribution(self,
                         state: torch.Tensor,
                         mask: torch.Tensor | None = None) -> torch.distributions.Distribution:
        """
        Full forward pass: State -> Features -> Logits -> Distribution.
        """
        # 1. Extract Features
        # shape: (Batch, Rows, Len, Hidden)
        features = self.backbone(state, mask)

        # 2. Project to Logits
        # shape: (Batch, Rows, Len, Output_Dim)
        logits = self.head(features)

        # 3. Create Distribution via Adapter
        # The adapter handles the masking logic (force mean=0 on padding)
        dist = self.adapter.logits_to_dist(logits, attention_mask=mask)

        return dist

    def get_action(self,
                   state: torch.Tensor,
                   mask: torch.Tensor | None = None,
                   deterministic: bool = False) -> torch.Tensor:
        """
        Inference helper. Returns the raw action tensor (integers or floats).
        """
        dist = self.get_distribution(state, mask)

        if deterministic:
            # For Normal distribution, 'mean' is the mode.
            # For Categorical, we would need argmax.
            # We assume the distribution object has a 'mean' or 'mode' property,
            # or we rely on the logic that for Gaussian, sample() is centered anyway.
            if hasattr(dist, 'mean'):
                return dist.mean
            elif hasattr(dist, 'mode'):
                return dist.mode
            else:
                # Fallback for distributions without explicit mode (should not happen with standard ones)
                return dist.sample()
        else:
            return dist.sample()