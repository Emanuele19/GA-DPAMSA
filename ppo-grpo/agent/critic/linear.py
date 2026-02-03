import torch
from torch import nn
from ..interfaces import IMSACritic, IMSABackbone

class MSA_Critic(IMSACritic):
    """
    Concrete implementation of the Value Function (Critic).
    It estimates the expected score (SP/CS) of the sub-board state.
    """

    def __init__(self, backbone: IMSABackbone):
        super().__init__(backbone)

        # Architecture:
        # 1. Masked Global Pooling (collapses spatial dims)
        # 2. MLP Head (Hidden -> 64 -> 1)
        self.value_head = nn.Sequential(
            nn.Linear(backbone.output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)  # Outputs a single scalar value
        )

    def get_value(self,
                  state: torch.Tensor,
                  mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Estimates V(s).
        Returns: Tensor of shape (Batch, 1)
        """
        # 1. Extract Features
        # shape: (Batch, Rows, Len, Hidden)
        features = self.backbone(state, mask)

        # 2. Masked Global Pooling
        # We need to aggregate information from all rows and columns into one vector.
        # Simple .mean() is bad because it includes padding zeros.

        if mask is not None:
            # Expand mask to match feature dimension
            # mask: (B, R, L) -> (B, R, L, 1)
            mask_expanded = mask.unsqueeze(-1).float()

            # Zero out features at padding positions (just in case backbone didn't)
            features = features * mask_expanded

            # Sum over Rows and Length (dim 1 and 2)
            sum_features = features.sum(dim=[1, 2])  # (Batch, Hidden)

            # Count valid tokens
            # sum over Rows and Length
            valid_tokens = mask_expanded.sum(dim=[1, 2])  # (Batch, 1)

            # Avoid division by zero
            valid_tokens = torch.clamp(valid_tokens, min=1.0)

            # Compute Average
            pooled_features = sum_features / valid_tokens

        else:
            # Fallback if no mask provided (assume full valid)
            pooled_features = features.mean(dim=[1, 2])

        # 3. Value Projection
        value = self.value_head(pooled_features)

        return value