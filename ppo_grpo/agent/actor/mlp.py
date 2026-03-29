import torch
import torch.nn as nn
import torch.nn.functional as F
from ..interfaces import IMSAActor, IMSABackbone, IMSAOutputAdapter


class MSA_Actor(IMSAActor):
    """
    Robust Actor Implementation.
    Uses an MLP Head instead of a single Linear layer to decouple
    feature extraction from policy decision making.
    """

    def __init__(self, backbone: IMSABackbone, adapter: IMSAOutputAdapter):
        super().__init__(backbone, adapter)

        hidden_dim = backbone.output_dim
        output_dim = adapter.logit_dim


        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

        # self._init_last_layer()

    def _init_last_layer(self):
        """
        Forces the model to start in a 'Safe Exploration' zone.
        """
        last_layer = self.head[-1]

        nn.init.uniform_(last_layer.weight, -0.01, 0.01)


        nn.init.constant_(last_layer.bias[0], 0.5)

        if last_layer.bias.shape[0] > 1:
            nn.init.constant_(last_layer.bias[1], 0.5)

    def get_distribution(self,
                         state: torch.Tensor,
                         mask: torch.Tensor | None = None) -> torch.distributions.Distribution:
        """
        Full forward pass: State -> Features -> MLP Head -> Distribution.
        """

        features = self.backbone(state, mask)

        logits = self.head(features)

        dist = self.adapter.logits_to_dist(logits, attention_mask=mask)

        return dist

    def get_action(self,
                   state: torch.Tensor,
                   mask: torch.Tensor | None = None,
                   deterministic: bool = False) -> torch.Tensor:

        dist = self.get_distribution(state, mask)

        if deterministic:
            return dist.mean
        else:
            return dist.sample()