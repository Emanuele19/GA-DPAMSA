import torch
import torch.nn as nn
import torch.nn.functional as F

class MSABranchingNet(nn.Module):
    def __init__(self, n_sequences, vocab_size, embedding_dim=16, hidden_dim=128):
        super(MSABranchingNet, self).__init__()
        self.n_sequences = n_sequences
        
        # hared Encoder
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
        
        # 2. Single head for value state estimation
        self.value_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 3. N heads for advantage function estimation
        self.advantage_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2) # [Base, Gap]
            ) for _ in range(n_sequences)
        ])

    def forward(self, x):
        # x shape: (Batch, N, 30)
        x = x.long()
        x = self.embedding(x).permute(0, 3, 1, 2) # (B, Emb, N, 30)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Slicing and Flattening
        # x[:, :, :, 0] -> (B, 128, N) -> reshape -> (B, 128*N)
        feat = x[:, :, :, 0].reshape(x.size(0), -1)
        
        v = self.value_head(feat)
        
        # A_i(s, a) -> (B, N, 2)
        # Stack the results of every branch
        a = torch.stack([head(feat) for head in self.advantage_heads], dim=1)
        
        # Value and Advantage aggregation
        # Q_i = V + (A_i - mean(A_i)) | "+" is broadcasted because V is scalar and A_i is vector
        # v.unsqueeze(2) makes v's shape from (B, 1) to (B, 1, 1) so it can be summed to a's shape (B, N, 2)
        q_values = v.unsqueeze(2) + (a - a.mean(dim=2, keepdim=True))
        
        return q_values # Tensor (Batch, N_seqs, 2)