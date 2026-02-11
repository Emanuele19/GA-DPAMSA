import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from ..interfaces import IMSABackbone


class DCNNBackbone(IMSABackbone):
    """
    A Dilated Convolutional Neural Network (DCNN) backbone optimized for MSA.

    Architecture Philosophy:
    1. Independent Sequence Processing: Uses (1, K) kernels to extract
       temporal features (k-mers, motifs) from each sequence independently.
    2. Global Column Mixing: Uses a (Rows, 1) kernel to aggregate information
       across all sequences for a specific column. This allows the network
       to 'see' the vertical alignment quality.
    """

    def __init__(self,
                 num_rows: int = config.AGENT_WINDOW_ROW,
                 vocab_size: int = config.VOCAB_SIZE,
                 embedding_dim: int = 64,
                 hidden_dim: int = 128):
        """
        Args:
            num_rows: Number of sequences in the sub-board (fixed height).
            vocab_size: Size of the vocabulary (A, C, G, T, -, ...).
            embedding_dim: Dimension of the input embedding.
            hidden_dim: Number of channels in the convolutional layers.
        """
        super().__init__()
        self._output_dim = hidden_dim

        # 1. Embedding Layer
        # Maps integer indices to dense vectors.
        # Padding index 0 is ignored (masked out).
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 2. DCNN Layers (Temporal Feature Extraction)
        # These layers look HORIZONTALLY (along the sequence).
        # Kernel (1, 3) means: Look at 1 row, 3 columns.
        # Dilation expands the receptive field without losing resolution.
        self.conv1 = nn.Conv2d(embedding_dim, 64, kernel_size=(1, 3),
                               dilation=(1, 1), padding=(0, 1))

        self.conv2 = nn.Conv2d(64, hidden_dim, kernel_size=(1, 3),
                               dilation=(1, 2), padding=(0, 2))

        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3),
                               dilation=(1, 4), padding=(0, 4))

        # 3. Vertical Mixing Layer (The "Alignment" Layer)
        # This layer looks VERTICALLY (across sequences).
        # Kernel (num_rows, 1) means: Look at ALL rows, 1 column.
        # Groups=1 ensures fully connected mixing of channels.
        self.col_mixer = nn.Conv2d(hidden_dim, hidden_dim,
                                   kernel_size=(num_rows, 1),
                                   padding=0)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (Batch, Rows, Max_Len) of Long integers.
            mask: Optional attention mask (ignored for DCNN usually, handled by padding).

        Returns:
            Tensor of shape (Batch, Rows, Max_Len, Hidden_Dim).
        """
        # x shape: (Batch, Rows, Len)

        # 1. Embedding
        # Output: (Batch, Rows, Len, Emb_Dim)
        x_emb = self.embedding(x)

        # Permute for Conv2d: (Batch, Channels, Height=Rows, Width=Len)
        x_conv = x_emb.permute(0, 3, 1, 2)

        # 2. Horizontal Processing (Independent per row)
        x_conv = F.relu(self.conv1(x_conv))
        x_conv = F.relu(self.conv2(x_conv))
        x_conv = F.relu(self.conv3(x_conv))
        # Shape: (Batch, Hidden_Dim, Rows, Len)

        # 3. Vertical Processing (Mixing across rows)
        # We perform a column-wise convolution.
        # Since kernel height == Rows, the resulting height will be 1.
        # Shape: (Batch, Hidden_Dim, 1, Len)
        global_col_features = F.relu(self.col_mixer(x_conv))

        # 4. Residual Connection / Broadcasting
        # We add the global column info back to the local row features.
        # This tells every nucleotide: "Here is what the whole column looks like".
        # We expand (Batch, Hidden, 1, Len) to match (Batch, Hidden, Rows, Len)
        out = x_conv + global_col_features.expand_as(x_conv)

        # 5. Restore format for the Actor
        # Target: (Batch, Rows, Len, Hidden_Dim)
        out = out.permute(0, 2, 3, 1)

        return out