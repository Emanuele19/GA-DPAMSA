import math
import os
import torch
import random

"""
Configuration File

This script defines the configuration settings for the GA-DPAMSA framework, including:
- Hyperparameters for Deep Q-Network (DQN) and Genetic Algorithm (GA).
- File paths for datasets, model weights, and results.
- Setup for external multiple sequence alignment (MSA) tools.
- Automatic directory creation to ensure required folders exist.

Author: https://github.com/ZhangLab312/DPAMSA
Co-Author: https://github.com/FLaTNNBio/GA-DPAMSA
"""

# ===========================
# Nucleotide Encoding
# ===========================
GAP_CHARACTER = '-'
PAD_CHARACTER = 'P'
NUCLEOTIDE_ENCODING = {PAD_CHARACTER: 0, 'A': 1, 'T': 2, 'C': 3, 'G': 4, GAP_CHARACTER: 5, 'N': 6,
                       PAD_CHARACTER.lower(): 0, 'a': 1, 't': 2, 'c': 3, 'g': 4, GAP_CHARACTER: 5, 'n': 6}

NUCLEOTIDE_DECODING = {0: PAD_CHARACTER, 1: 'A', 2: 'T', 3: 'C', 4: 'G', 5: GAP_CHARACTER}



# ===========================
# DPAMSA Hyperparameters
# ===========================
GAP_PENALTY = -2  # Penalty for inserting a gap
MISMATCH_PENALTY = -3  # Penalty for a mismatch
MATCH_REWARD = 6  # Reward for a correct match
TIME_PENALTY = -0.2 # Penalty for time passing. To avoid stalls.
MICROTIME_PENALTY = TIME_PENALTY / 10 # Changes the Q values by a small amount to avoid persistent actions
PENALTY_MULTIPLIER = 1.5 # For VERY bad actions
MAX_EPISODE = 19000  # Maximum number of training episodes
BATCH_SIZE = 64  # Number of experiences sampled per training step
REPLAY_MEMORY_SIZE = 1000  # Capacity of replay memory buffer
ALPHA = 1e-5  # Learning rate for the optimizer
EPSILON = 0.95  # Initial epsilon value for ε-greedy policy
MIN_EPSILON = 0.2  # Minimum epsilon value
GAMMA = 0.95  # Discount factor for Q-learning
EPSILON_DECAY = 0.9998  # Decay rate for epsilon (e_t = e_{t-1} * decay)
UPDATE_ITERATION = 1000  # Number of iterations before updating the target network
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # Auto-detect GPU or CPU

# ===========================
# Random Seed Configuration
# ==========================
SEED = 42  
rng = random.Random(SEED)

# ===========================
# Genetic Algorithm (GA) Parameters
# ===========================
AGENT_WINDOW_ROW = 3  # Number of rows in the agent's observation window
AGENT_WINDOW_COLUMN = 30 # Number of columns in the observation window
GA_ITERATIONS = 3  # Number of iterations for genetic evolution
POPULATION_SIZE = 5  # Population size for genetic algorithm
CLONE_RATE = 0.25  # % of the population to be an exact copy of the input sequences during Population Generation Phase
GAP_RATE = 0.05  # % of Gap to be added to an individual during Population Generation Phase (calculated on seq. length)
SELECTION_RATE = 0.5  # % of the population to be selected following a certain criteria
MUTATION_RATE = 0.25  # % of the population undergo mutation


# Ensure hyperparameter constraints
assert 0 < BATCH_SIZE <= REPLAY_MEMORY_SIZE, "batch size must be in the range of 0 to the size of replay memory."
assert ALPHA > 0, "alpha must be greater than 0."
assert 0 <= GAMMA <= 1, "gamma must be in the range of 0 to 1."
assert 0 <= EPSILON <= 1, "epsilon must be in the range of 0 to 1."


# ===========================
# File Paths Configuration
# ===========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset Paths
BASE_DATASETS_PATH = os.path.join(PROJECT_ROOT, "datasets")
FASTA_FILES_PATH = os.path.join(BASE_DATASETS_PATH, "fasta_files")
TRAINING_DATASET_PATH = os.path.join(BASE_DATASETS_PATH, "training_dataset")
INFERENCE_DATASET_PATH = os.path.join(BASE_DATASETS_PATH, "inference_dataset")

# Model Weights Path
DPAMSA_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "DPAMSA", "weights")
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'DCNNMSA', 'weights') # For new models

# Tensorboard Training Runs Path
RUNS_PATH = os.path.join(PROJECT_ROOT, "DPAMSA", "runs")

# Results Paths
BASE_RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")
REPORTS_PATH = os.path.join(BASE_RESULTS_PATH, "reports")
DPAMSA_REPORTS_PATH = os.path.join(REPORTS_PATH, "DPAMSA")
GA_DPAMSA_REPORTS_PATH = os.path.join(REPORTS_PATH, "GA-DPAMSA")
DATASETS_REPORTS_PATH = os.path.join(REPORTS_PATH, "datasets")
BENCHMARKS_PATH = os.path.join(BASE_RESULTS_PATH, "benchmarks")
TOOLS_OUTPUT_PATH = os.path.join(BASE_RESULTS_PATH, "tools_output")
CSV_PATH = os.path.join(BENCHMARKS_PATH, "csv")
DATASETS_CSV_PATH = os.path.join(CSV_PATH, "datasets")
INFERENCE_CSV_PATH = os.path.join(CSV_PATH, "inference")
DPAMSA_INF_CSV_PATH = os.path.join(INFERENCE_CSV_PATH, "DPAMSA")
GA_DPAMSA_INF_CSV_PATH = os.path.join(INFERENCE_CSV_PATH, "GA-DPAMSA")
CHARTS_PATH = os.path.join(BENCHMARKS_PATH, "charts")

# Ensure directories exist, creating them if they don't
REQUIRED_DIRECTORIES = [
    DPAMSA_WEIGHTS_PATH,
    RUNS_PATH,
    BASE_RESULTS_PATH,
    DPAMSA_REPORTS_PATH,
    GA_DPAMSA_REPORTS_PATH,
    DATASETS_REPORTS_PATH,
    BENCHMARKS_PATH,
    TOOLS_OUTPUT_PATH,
    CSV_PATH,
    DATASETS_CSV_PATH,
    INFERENCE_CSV_PATH,
    DPAMSA_INF_CSV_PATH,
    GA_DPAMSA_INF_CSV_PATH,
    CHARTS_PATH
]
for path in REQUIRED_DIRECTORIES:
    if not os.path.exists(path):
        os.makedirs(path)


# ===========================
# External MSA Tools Configuration
# ===========================
TOOLS = {
    'ClustalOmega': {
        'command': lambda file_path, output_dir: ['clustalo', '-i', file_path, '-o', output_dir],
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'ClustalOmega'),
        'report_dir': os.path.join(REPORTS_PATH, 'ClustalOmega')
    },
    'MSAProbs': {
        'command': lambda file_path, output_dir: ['msaprobs', file_path, '-o', output_dir],
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'MSAProbs'),
        'report_dir': os.path.join(REPORTS_PATH, 'MSAProbs')
    },
    'ClustalW': {
        'command': lambda file_path, output_dir: ['clustalw', f'-INFILE={file_path}',
                                                  '-OUTPUT=FASTA', f'-OUTFILE={output_dir}'],
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'ClustalW'),
        'report_dir': os.path.join(REPORTS_PATH, 'ClustalW')
    },
    'MAFFT': {
        'command': lambda file_path, output_dir: f"mafft --auto {file_path} > {output_dir}",
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'MAFFT'),
        'report_dir': os.path.join(REPORTS_PATH, 'MAFFT')
    },
    'MUSCLE5': {
        'command': lambda file_path, output_dir: ['muscle5', '-align', file_path, '-output', output_dir],
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'MUSCLE5'),
        'report_dir': os.path.join(REPORTS_PATH, 'MUSCLE5')
    },
    'UPP': {
        'command': lambda file_path, output_dir: ['run_upp.py', '-s', file_path, '-m', 'dna', '-d', output_dir],
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'UPP'),
        'report_dir': os.path.join(REPORTS_PATH, 'UPP')
    },
    'PASTA': {
        'command': lambda file_path, output_dir: ['run_pasta.py', '-i', file_path, '-o', output_dir],
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'PASTA'),
        'report_dir': os.path.join(REPORTS_PATH, 'PASTA')
    }
}

# ===========================
# Models Configuration
# ===========================
MODELS = {
    "GA-DPAMSA": {
        "model_path": "model_3x30",
        "report_path": GA_DPAMSA_REPORTS_PATH,
        "csv_path": GA_DPAMSA_INF_CSV_PATH,
        "requires_dataset_object": True
    },
    "DPAMSA": {
        "model_path": "model_3x30",
        "report_path": DPAMSA_REPORTS_PATH,
        "csv_path": DPAMSA_INF_CSV_PATH,
        "requires_dataset_object": True
    },
    "DCNNMSA": {
        "model_path": os.path.join(PROJECT_ROOT, 'DCNNMSA', 'weights', 'msa_model_ep18999.pth'),
        "report_path": os.path.join(REPORTS_PATH, 'DCNNMSA'),
        "csv_path": os.path.join(INFERENCE_CSV_PATH, 'DCNNMSA'),
        "requires_dataset_object": False
    },
    "DCNN_BDDQNMSA": {
        "model_path": os.path.join(PROJECT_ROOT, 'DCNN_BDDQNMSA', 'weights', 'msa_model_ep18999.pth'),
        "report_path": os.path.join(REPORTS_PATH, 'DCNN_BDDQNMSA'),
        "csv_path": os.path.join(INFERENCE_CSV_PATH, 'DCNN_BDDQNMSA'),
        "requires_dataset_object": False
    }
}
