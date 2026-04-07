import torch
import torch.nn as nn
import torch.nn.functional as F

class MSABranchingNet(nn.Module):
    def __init__(self, n_sequences, vocab_size, embedding_dim=16, hidden_dim=256):
        super(MSABranchingNet, self).__init__()
        self.n_sequences = n_sequences
        
        # Shared Encoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Kernel height and width
        kh = n_sequences if n_sequences % 2 != 0 else n_sequences + 1
        kw = 3

        # Fixed vertical padding
        pad_h = (kh - 1) // 2

        # Dilations
        d1, d2, d3 = 1, 2, 4

        self.conv1 = nn.Conv2d(embedding_dim, 32, 
                               kernel_size=(kh, kw), dilation=(1, d1), padding=(pad_h, d1))
        self.conv2 = nn.Conv2d(32, 64, 
                               kernel_size=(kh, kw), dilation=(1, d2), padding=(pad_h, d2))
        self.conv3 = nn.Conv2d(64, 128, 
                               kernel_size=(kh, kw), dilation=(1, d3), padding=(pad_h, d3))
        
        # Post flattening dimension
        self.feature_dim = 128 * n_sequences
        
        # Shared compressed information
        # Reduces dimensionality from 1280 to 256
        self.compression = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # The advantage head has been implemented as a unique block to
        # allow parallel processing.
        self.advantage_net = nn.Linear(hidden_dim, n_sequences * 2)

    def forward(self, x):
        # x shape: (Batch, N, L)
        batch_size = x.size(0)
        
        # Encoder
        x = x.long()
        x = self.embedding(x).permute(0, 3, 1, 2) # (B, Emb, N, L)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Slicing and Flattening
        # x[:, :, :, 0] -> (B, 128, N) -> reshape -> (B, 128*N)
        feat = x[:, :, :, 0].reshape(batch_size, -1)
        
        # Shared compressed information
        compressed = self.compression(feat) # (B, 256)
        
        v = self.value_head(compressed) # (B, 1)
        
        a = self.advantage_net(compressed) 
        a = a.view(batch_size, self.n_sequences, 2) 
        
        #  Value and Advantage aggregation
        # Q_i = V + (A_i - mean(A_i)) | "+" is broadcasted because V is scalar and A_i is vector
        # v.unsqueeze(2) makes v's shape from (B, 1) to (B, 1, 1) so it can be summed to a's shape (B, N, 2)
        q_values = v.unsqueeze(2) + (a - a.mean(dim=2, keepdim=True))
        
        return q_values # Shape finale: (Batch, N, 2)