import os
from tqdm import tqdm

import config


"""
DNA Sequence Dataset Generator

This script generates synthetic DNA sequence datasets with conserved blocks, mutations, and gaps. 
Users can configure various parameters such as sequence length, number of sequences, mutation rates, 
and whether to mutate inside conserved blocks. The generated datasets are stored in FASTA format, 
and a Python script is created to structure the dataset information.

Usage:
- Modify the configuration parameters in the script header to adjust dataset properties.
- Run the script to generate the datasets.

Output:
- A set of FASTA files containing the generated sequences.
    (Suited for tools benchmarking)
    
- A Python file with all the sets of generated sequences PLUS some metadata about the dataset itself.
    (Suited for both training and inference on DPAMSA and for GA-DPAMSA inference)
"""

# ================= CONFIGURATION ================= #
# User-configurable parameters
num_sequences = 3            # Number of DNA sequences per dataset
sequence_length = 30         # Length of each sequence
mutation_rate = 0.10         # Mutation probability (10%)
gap_rate = 0.05              # Gap insertion probability (5%)
number_of_datasets = 50      # Total number of datasets to generate
min_score_threshold = 0     # Minimum alignment score threshold
max_score_threshold = None   # Maximum alignment score threshold (None = no limit)

# Conserved block settings
num_conserved_blocks = 1      # Number of conserved blocks per sequence
conserved_block_sizes = [18]  # List of block sizes (one size per block)

# Additional options
fixed_block_position = False  # True = fixed position, False = random position
mutate_inside_blocks = False  # True = mutations inside blocks allowed, False = only outside

# File paths
DATASET_NAME = 'random_dataset'
FASTA_OUTPUT = os.path.join(config.FASTA_FILES_PATH, DATASET_NAME)
PY_OUTPUT = os.path.join(config.INFERENCE_DATASET_PATH, f'{DATASET_NAME}.py')

# Create output directories if they do not exist
if not os.path.exists(FASTA_OUTPUT):
    os.makedirs(FASTA_OUTPUT)


# ================= FUNCTION DEFINITIONS ================= #
def generate_random_dna_sequence(length, rng):
    """Generate a random DNA sequence of a given length (without gaps)."""
    return ''.join(rng.choice("ATCG") for _ in range(length))


def mutate_sequence(sequence, mutation_rate, conserved_blocks, insert_positions, rng):
    """
    Introduce mutations into the DNA sequence.

    Mutations occur at random positions outside conserved blocks unless configured otherwise.

    Parameters:
    ----------
    - sequence (str): The original DNA sequence.
    - mutation_rate (float): Probability of mutation per base.
    - conserved_blocks (list): List of conserved sequences.
    - insert_positions (list): Corresponding positions of conserved blocks.

    Returns:
    -------
    - str: The mutated DNA sequence.
    """
    mutated_sequence = list(sequence)
    for i in range(len(sequence)):
        in_conserved = any(insert <= i < insert + len(block) for insert, block in zip(insert_positions, conserved_blocks))
        if mutate_inside_blocks or not in_conserved:
            if rng.random() < mutation_rate:
                mutated_sequence[i] = rng.choice("ATCG")
    return ''.join(mutated_sequence)


def insert_random_gaps(sequence, gap_rate, conserved_blocks, insert_positions, rng, max_gaps=None):
    """
    Insert random gaps into a DNA sequence, avoiding conserved blocks.

    Parameters:
    -----------
    - sequence (str): The DNA sequence to modify.
    - gap_rate (float): Probability of inserting a gap per base.
    - conserved_blocks (list): Conserved sequences.
    - insert_positions (list): Block insertion positions.
    - max_gaps (int, optional): Maximum number of gaps to insert.

    Returns:
    --------
    - str: The DNA sequence with gaps inserted.
    """
    gapped_sequence = list(sequence)
    gap_count = 0
    for i in range(len(sequence)):
        in_conserved = any(insert <= i < insert + len(block) for insert, block in zip(insert_positions, conserved_blocks))
        if not in_conserved:
            if rng.random() < gap_rate:
                gapped_sequence[i] = '-'
                gap_count += 1
                if max_gaps and gap_count >= max_gaps:
                    break
    return ''.join(gapped_sequence)


def calculate_alignment_score(sequences):
    """
    Calculate an alignment score based on pairwise comparisons of sequences.

    Scoring system:
    --------------
    - Matching bases: +4
    - Mismatches or gaps: -4

    Parameters:
    ----------
    - sequences (list): List of DNA sequences.

    Returns:
    -------
    - int: Alignment score.
    """
    score = 0
    for k in range(len(sequences[0])):
        for i in range(len(sequences) - 1):
            for j in range(i + 1, len(sequences)):
                x, y = sequences[i][k], sequences[j][k]
                score += 4 if x == y else -4
    return score


def generate_dataset(seq_length, conserved_block_sizes, mutation_rate, gap_rate, min_score_threshold, max_score_threshold, rng):
    """
    Generate a dataset of sequences containing conserved blocks with mutations and gaps.

    Returns a dataset only if the alignment score is within the specified range.

    Returns:
    -------
    - list: The generated DNA sequences.
    - list: The conserved blocks used.
    - list: Positions where blocks were inserted.
    - int: The final alignment score.
    """
    while True:
        conserved_blocks = [generate_random_dna_sequence(size, rng) for size in conserved_block_sizes]
        insert_positions = [rng.randint(0, seq_length - len(block)) for block in conserved_blocks] if not fixed_block_position else [5] * len(conserved_blocks)

        sequences = []
        for _ in range(num_sequences):
            sequence = generate_random_dna_sequence(seq_length, rng)
            for i, block in enumerate(conserved_blocks):
                sequence = sequence[:insert_positions[i]] + block + sequence[insert_positions[i] + len(block):]
            sequence = mutate_sequence(sequence, mutation_rate, conserved_blocks, insert_positions, rng)
            sequence = insert_random_gaps(sequence, gap_rate, conserved_blocks, insert_positions, rng)
            sequences.append(sequence)

        score = calculate_alignment_score(sequences)
        if min_score_threshold <= score and (max_score_threshold is None or score <= max_score_threshold):
            return sequences, conserved_blocks, insert_positions, score


def write_fasta_file(filename, sequences):
    """Write DNA sequences to a FASTA file."""
    with open(filename, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">Sequence_{i+1}\n{seq}\n")


def write_dataset_file(fasta_files, output_file):
    """Generate a Python file containing structured dataset information."""
    datasets = {}
    for fasta in fasta_files:
        sequences = {}
        with open(fasta, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    seq_name = line.strip().lstrip('>')
                    sequences[seq_name] = ""
                else:
                    sequences[seq_name] += line.strip()
        dataset_name = os.path.splitext(os.path.basename(fasta))[0]  # Remove .fasta extension
        datasets[dataset_name] = sequences

    with open(output_file, 'w') as f:
        f.write(f"file_name = '{os.path.basename(output_file)}'\n\ndatasets = {list(datasets.keys())}\n")
        f.write(f"SEED = {config.SEED}\n\n")
        for name, seqs in datasets.items():
            f.write(f"\n{name} = {list(seqs.values())}\n")

# ================= Additional Dataset Generation Functions ================= #

def generate_fully_random_dataset(num_sequences, sequence_length, rng):
    """
    Generate a dataset of completely random DNA sequences.
    
    - No conserved blocks
    - No implicit alignment
    - Fully reproducible via rng
    
    Parameters:
    ----------
    - num_sequences (int): number of sequences
    - sequence_length (int): length of each sequence
    - rng (random.Random): local random generator
    
    Returns:
    -------
    - list[str]: list of random DNA sequences
    """
    sequences = []
    for _ in range(num_sequences):
        seq = ''.join(rng.choice("ATCG") for _ in range(sequence_length))
        sequences.append(seq)
    return sequences


def generate_fully_random_dataset(num_sequences, sequence_length, rng):
    return [
        ''.join(rng.choice("ATCG") for _ in range(sequence_length))
        for _ in range(num_sequences)
    ]


# ================= EXECUTION ================= #
if __name__ == "__main__":
    fasta_files = []
    for dataset_idx in tqdm(range(number_of_datasets), desc="Generating datasets"):
        #sequences, conserved_blocks, insert_positions, score = generate_dataset(sequence_length, conserved_block_sizes, mutation_rate, gap_rate, min_score_threshold, max_score_threshold)
        sequences = generate_fully_random_dataset(
            num_sequences, sequence_length, config.rng)
        # print(f"Dataset {dataset_idx}: Conserved blocks {conserved_blocks} at positions {insert_positions}, "
        #      f"score {score}")

        fasta_filename = os.path.join(FASTA_OUTPUT, f'random_{dataset_idx}.fasta')
        write_fasta_file(fasta_filename, sequences)
        fasta_files.append(fasta_filename)

    write_dataset_file(fasta_files, PY_OUTPUT)
    print(f"Random saved in {PY_OUTPUT}")
