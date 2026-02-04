# --- config.py REQUIRED ADDITIONS ---

# Algorithm Selection: 'GRPO' or 'PPO'
ALGO = 'GRPO'

# Dimensions
AGENT_WINDOW_ROW = 3       # Number of sequences in a sub-board (block)
BATCH_SIZE = 4             # Number of sub-boards to process at once
MAX_GAPS_PER_POS = 5       # Maximum gaps the model can predict per position

# Training Hyperparameters
LR = 1e-4                  # Learning Rate
MAX_EPOCH = 100            # Total training epochs
SAVE_FREQ = 10             # Save weights every N epochs
GRPO_GROUP_SIZE = 8        # (Only for GRPO) Number of parallel samples per input

# Paths
FASTA_FILES_PATH = "./data_raw"
TENSORBOARD_PATH = "./runs/experiment_gen_01"
MODEL_WEIGHTS_PATH = "./weights"

# Symbols mapping
NUCLEOTIDES_MAP = {
    'A': 1, 'a': 1,
    'T': 2, 't': 2,
    'C': 3, 'c': 3,
    'G': 4, 'g': 4,
    '-': 5,
    'N': 6, 'n': 6,
    'P': 0, 'p' : 0
}
NUCLEOTIDES = ['P','A', 'T', 'C', 'G', '-', 'N']

GAP_TOKEN = 5

PADDING_TOKEN = 0

VOCAB_SIZE = len(NUCLEOTIDES)
