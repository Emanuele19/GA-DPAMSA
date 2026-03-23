import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config
from ..interfaces import IMSABackbone


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard Positional Encoding (come nei Transformer).
    Injects information about the absolute position of tokens.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape: (1, 1, Max_Len, D_Model) per broadcasting facile
        self.register_buffer('pe', pe.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        # x shape: (Batch, Rows, Len, Emb)
        # pe slice: (1, 1, Len, Emb)
        return x + self.pe[:, :, :x.size(2), :]


class AlignmentBlock(nn.Module):
    """
    A single block that processes both Horizontally (Sequence) and Vertically (Alignment).
    Uses Residual connections.
    """

    def __init__(self, channels, num_rows, dilation=1):
        super().__init__()

        # 1. Horizontal Processing (1D Conv treating Rows as separate channels/batches essentially)
        # Look at neighboring nucleotides
        self.conv_h = nn.Conv2d(channels, channels, kernel_size=(1, 3),
                                padding=(0, dilation), dilation=(1, dilation))

        # 2. Vertical Processing (Column Mixing)
        # Look at the whole column to check consensus/mismatches
        # Kernel (Rows, 1) collapses the rows, but we keep dimensions to broadcast back
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=(num_rows, 1), padding=0)

        self.norm = nn.LayerNorm(channels)  # LayerNorm is usually better for NLP/Bio than BatchNorm

    def forward(self, x):
        # x shape: (Batch, Channels, Rows, Len)
        residual = x

        # Horizontal Step
        out = F.relu(self.conv_h(x))

        # Vertical Step (Global Column Context)
        # v shape: (Batch, Channels, 1, Len)
        v = self.conv_v(out)

        # Add Vertical info back to Horizontal info (Broadcasting over Rows)
        out = out + v.expand_as(out)

        # Residual Connection
        out = out + residual

        # Normalization (Requires permuting because LayerNorm expects channels last)
        # (B, C, R, L) -> (B, R, L, C)
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = F.relu(out)
        # Back to (B, C, R, L)
        out = out.permute(0, 3, 1, 2)

        return out


class RobustBackbone(IMSABackbone):
    """
    Improved Backbone with Positional Encodings and Iterative Mixing.
    """

    def __init__(self,
                 num_rows: int = config.AGENT_WINDOW_ROW,
                 vocab_size: int = config.VOCAB_SIZE,
                 embedding_dim: int = 128,  # Aumentato per capacità
                 hidden_dim: int = 128,
                 num_layers: int = 4):  # Più profondo
        super().__init__()
        self._output_dim = hidden_dim

        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 2. Positional Encoding (CRITICO!)
        self.pos_encoder = SinusoidalPositionalEncoding(embedding_dim, max_len=config.AGENT_WINDOW_COLUMN + 50)

        # 3. Projection to Hidden size
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)

        # 4. Stack of Alignment Blocks
        self.layers = nn.ModuleList([
            AlignmentBlock(hidden_dim, num_rows, dilation=2 ** i)
            for i in range(num_layers)
        ])

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (Batch, Rows, Len)

        # Embed + Positional Encoding
        x = self.embedding(x)  # (B, R, L, Emb)
        x = self.pos_encoder(x)
        x = self.input_proj(x)  # (B, R, L, Hidden)

        # Permute for Conv2d: (Batch, Hidden, Rows, Len)
        x = x.permute(0, 3, 1, 2)

        # Pass through blocks
        for layer in self.layers:
            x = layer(x)

        # Final permute back to (Batch, Rows, Len, Hidden)
        x = x.permute(0, 2, 3, 1)

        return x