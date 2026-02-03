import torch.nn as nn
import torch.nn.functional as F

class MSANet(nn.Module):
    def __init__(self, n_sequences, vocab_size, embedding_dim=16, hidden_dim=128):
        super(MSANet, self).__init__()
        self.n_sequences = n_sequences
        self.n_actions = 2**n_sequences - 1
        
        # 1. Embedding Layer: trasforma indici (0-5) in vettori densi
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. DCNN Layers (Padding='same' per mantenere la larghezza 30)
        # Input shape per Conv2d: (Batch, Channels, Height, Width) -> (B, embedding_dim, N, 30)
        self.conv1 = nn.Conv2d(embedding_dim, 32, kernel_size=(1, 3), dilation=(1, 1), padding=(0, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 3), dilation=(1, 2), padding=(0, 2))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 3), dilation=(1, 4), padding=(0, 4))
        
        # 3. MLP Head
        # Dopo lo slicing della prima colonna avremo un vettore di taglia: 128 (canali) * N (sequenze)
        self.fc1 = nn.Linear(128 * n_sequences, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.n_actions)

    def forward(self, x):
        x = x.long() # x is stored as tensor of uint8, but Net wants tensor of Long
        # x shape: (Batch, N, 30)
        
        # Embedding -> (Batch, N, 30, Emb) -> Permute to (Batch, Emb, N, 30)
        x = self.embedding(x).permute(0, 3, 1, 2)
        
        # DCNN pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) # Shape: (Batch, 128, N, 30)
        
        # --- SLICING DELLA PRIMA COLONNA ---
        # Prendiamo solo la colonna all'indice 0 della dimensione Width (30)
        # x[:, :, :, 0] -> Shape: (Batch, 128, N)
        x = x[:, :, :, 0]
        
        # Flatten per l'MLP
        x = x.reshape(x.size(0), -1) 
        
        # MLP pass
        x = F.relu(self.fc1(x))
        return self.fc2(x) # Restituisce i Q-values