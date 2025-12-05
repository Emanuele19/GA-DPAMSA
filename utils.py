import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import subprocess
from tqdm import tqdm

import config
from DPAMSA.env import Environment

"""
DPAMSA Utility Functions

This script provides various utilities for:
- Genetic Algorithm (GA) operations for selecting sub-boards.
- Benchmarking and evaluation of MSA methods.
- Running inference using GA-DPAMSA and DPAMSA models.
- Generating plots to visualize performance.

Author: https://github.com/FLaTNNBio/GA-DPAMSA
"""


# ===========================
# Genetic Algorithm (GA) Utilities
# ===========================
def get_sum_of_pairs(chromosome, from_row, to_row, from_column, to_column):
    """
    Compute the Sum-of-Pairs (SP) score for a given sub-board in an MSA alignment.

    The SP score evaluates how well nucleotides are aligned in a subsection of an
    MSA matrix by comparing all possible pairs of symbols within each column.

    Parameters:
    -----------
    - chromosome (list of lists): The MSA alignment matrix, where each row represents a sequence and each column represents an aligned position.
    - from_row (int): Starting row index of the sub-board.
    - to_row (int): Ending row index (exclusive) of the sub-board.
    - from_column (int): Starting column index of the sub-board.
    - to_column (int): Ending column index (exclusive) of the sub-board.

    Returns:
    --------
    - int: The computed Sum-of-Pairs (SP) score for the selected sub-board.

    Scoring System:
    ---------------
    - **Gap (`5`)** → Adds `config.GAP_PENALTY`
    - **Exact match** → Adds `config.MATCH_REWARD`
    - **Mismatch** → Adds `config.MISMATCH_PENALTY`

    Example:
    --------
    >>> chromosome = [
    ...     [1, 2, 3, 5],  # A, T, C, -
    ...     [1, 2, 4, 5],  # A, T, G, -
    ...     [1, 3, 3, 5]   # A, C, C, -
    ... ]
    >>> get_sum_of_pairs(chromosome, 0, 3, 0, 4)
    (Computes SP score for the entire matrix)
    """
    score = 0

    # Iterate over all columns in the sub-board
    for i in range(from_column, to_column):
        # Compare all sequence pairs in the column
        for j in range(from_row, to_row):
            for k in range(j + 1, to_row):
                if chromosome[j][i] == 5 or chromosome[k][i] == 5:
                    score += config.GAP_PENALTY  # Penalize gaps
                elif chromosome[j][i] == chromosome[k][i]:
                    score += config.MATCH_REWARD  # Reward matches
                elif chromosome[j][i] != chromosome[k][i]:
                    score += config.MISMATCH_PENALTY  # Penalize mismatches

    return score


def get_column_score(chromosome, from_row, to_row, from_column, to_column):
    """
    Compute the Column Score (CS) for a given sub-board in an MSA alignment.

    The CS metric evaluates how well nucleotides are aligned by measuring the fraction
    of columns where all sequences (rows) contain the same nucleotide.

    Parameters:
    -----------
    - chromosome (list of lists): The MSA alignment matrix, where each row represents
                                  a sequence and each column represents an aligned position.
    - from_row (int): Starting row index of the sub-board.
    - to_row (int): Ending row index (exclusive) of the sub-board.
    - from_column (int): Starting column index of the sub-board.
    - to_column (int): Ending column index (exclusive) of the sub-board.

    Returns:
    --------
    - float: The fraction of fully matched columns within the selected sub-board.
             If no columns are present, returns 0.

    Example:
    --------
    >>> chromosome = [
    ...     [1, 2, 3, 3],  # A, T, C, C
    ...     [1, 2, 3, 3],  # A, T, C, C
    ...     [1, 2, 3, 3]   # A, T, C, C
    ... ]
    >>> get_column_score(chromosome, 0, 3, 0, 4)
    1.0  # All columns are fully matched

    >>> chromosome = [
    ...     [1, 2, 3, 4],  # A, T, C, G
    ...     [1, 2, 3, 3],  # A, T, C, C
    ...     [1, 2, 3, 3]   # A, T, C, C
    ... ]
    >>> get_column_score(chromosome, 0, 3, 0, 4)
    0.75  # 3 out of 4 columns are fully matched
    """
    # Number of columns in the selected sub-board
    num_columns = to_column - from_column

    # Count the number of columns where all sequences (rows) have the same value
    uniform_columns = sum(
        1 for col in range(from_column, to_column)
        if all(chromosome[row][col] == chromosome[from_row][col] for row in range(from_row, to_row))
    )

    # Return fraction of fully matched columns (avoid division by zero)
    return uniform_columns / num_columns if num_columns > 0 else 0


def is_overlap(range1, range2):
    """
    Check if two sub-board regions overlap.

    This function determines whether two rectangular regions (sub-boards)
    intersect based on their row and column coordinates.

    Parameters:
    -----------
    - range1 (tuple): Coordinates of the first sub-board in the format (row_start, row_end, col_start, col_end).
    - range2 (tuple): Coordinates of the second sub-board in the format (row_start, row_end, col_start, col_end).

    Returns:
    --------
    - bool: True if the two sub-boards overlap, False otherwise.

    Example:
    --------
    >>> is_overlap((0, 3, 0, 3), (2, 5, 2, 5))
    True
    >>> is_overlap((0, 2, 0, 2), (3, 5, 3, 5))
    False
    """
    # Extract row and column boundaries
    from_row1, to_row1, from_column1, to_column1 = range1
    from_row2, to_row2, from_column2, to_column2 = range2

    # Check if the sub-boards overlap in both rows and columns
    overlap_row = from_row1 < to_row2 and to_row1 > from_row2
    overlap_column = from_column1 < to_column2 and to_column1 > from_column2

    return overlap_row and overlap_column


def check_overlap(new_range,used_ranges):
    """
    Check if a new sub-board range overlaps with any previously used ranges.

    This function is used to ensure that newly selected sub-boards do not
    overlap with already chosen sub-boards.

    Parameters:
    -----------
    - new_range (tuple): The new sub-board range in the format
                         (row_start, row_end, col_start, col_end).
    - used_ranges (list of tuples): List of previously used sub-board ranges,
                                     each in the format (row_start, row_end, col_start, col_end).

    Returns:
    --------
    - bool: True if `new_range` overlaps with any range in `used_ranges`, False otherwise.

    Example:
    --------
    >>> used_ranges = [(0, 3, 0, 3), (4, 7, 4, 7)]
    >>> check_overlap((2, 5, 2, 5), used_ranges)
    True  # Overlaps with (0, 3, 0, 3)

    >>> check_overlap((7, 9, 7, 9), used_ranges)
    False  # No overlap with any existing range
    """
    for existing_range in used_ranges:
        if is_overlap(new_range, existing_range):
            return True  # Overlap detected
    return False  # No overlap found


def get_all_different_sub_range(individual, m_prime=config.AGENT_WINDOW_ROW, n_prime=config.AGENT_WINDOW_COLUMN):
    """
    Generate all unique, non-overlapping sub-boards of fixed size from an MSA alignment.

    This function extracts subsections (sub-boards) of size (m_prime, n_prime)
    from the main alignment matrix while ensuring:
    - No overlapping sub-boards.
    - Only valid sub-boards within sequence boundaries are selected.
    - Sequences with different lengths are handled by considering the shortest sequence.

    Parameters:
    -----------
    - individual (list of lists): The MSA alignment matrix, where each row represents
                                  a sequence and each column represents an aligned position.
    - m_prime (int): Number of rows in the sub-board (agent window row size).
    - n_prime (int): Number of columns in the sub-board (agent window column size).

    Returns:
    --------
    - list of tuples: A list of non-overlapping sub-boards, each represented as
                      (from_row, to_row, from_column, to_column).

    Example:
    --------
    >>> individual = [
    ...     [1, 2, 3, 4, 5],  # A, T, C, G, -
    ...     [1, 2, 3, 4, 5],  # A, T, C, G, -
    ...     [1, 2, 3, 4, 5]   # A, T, C, G, -
    ... ]
    >>> get_all_different_sub_range(individual, 2, 2)
    [(0, 2, 0, 2), (0, 2, 2, 4), (1, 3, 0, 2), (1, 3, 2, 4)]
    """
    m = len(individual)  # Total number of sequences (rows)

    # Find the shortest sequence length to handle variable-length sequences
    n = min(len(genes) for genes in individual)

    # List to store unique, non-overlapping sub-boards
    unique_ranges = []

    # Iterate over all possible starting positions for sub-boards
    for i in range(m):
        for j in range(n):
            from_row = i
            to_row = i + m_prime
            from_column = j
            to_column = j + n_prime

            # Ensure the sub-board does not overlap with previously selected ones
            if not check_overlap((from_row, to_row, from_column, to_column), unique_ranges):
                # Ensure the sub-board is within valid sequence boundaries
                if to_row <= m and to_column <= n:
                    unique_ranges.append((from_row, to_row, from_column, to_column))
    
    return unique_ranges


def calculate_worst_fitted_sub_board(individual, mode):
    """
    Identifies the worst-performing sub-board (sub-region) within an individual.

    The worst sub-board is the region with the lowest fitness score. Depending on the mode,
    this can be determined using:
    - 'sp': Sum-of-Pairs (SP) score.
    - 'cs': Column Score (CS).
    - 'mo': A normalized combination of SP and CS.

    Parameters:
    -----------
        individual (list of lists): The full MSA alignment matrix.
        mode (str): Evaluation mode ('sp', 'cs', or 'mo').

    Returns:
    --------
        tuple: (worst_score, (from_row, to_row, from_column, to_column))
               where worst_score is the lowest fitness score and the coordinates define the worst sub-board.
    """
    # Get all possible non-overlapping sub-boards
    unique_ranges = get_all_different_sub_range(individual, config.AGENT_WINDOW_ROW, config.AGENT_WINDOW_COLUMN)
    sub_board_scores = []

    sp_scores, cs_scores = [], []

    # Compute SP and CS scores for each sub-board
    for from_row, to_row, from_column, to_column in unique_ranges:
        sp_score = get_sum_of_pairs(individual, from_row, to_row, from_column, to_column)
        cs_score = get_column_score(individual, from_row, to_row, from_column, to_column)

        sp_scores.append(sp_score)
        cs_scores.append(cs_score)

        sub_board_scores.append((sp_score, cs_score, (from_row, to_row, from_column, to_column)))

    if mode == 'sp':
        # Select the sub-board with the lowest Sum-of-Pairs score
        worst_subboard = min(sub_board_scores, key=lambda x: x[0])
    elif mode == 'cs':
        # Select the sub-board with the lowest Column Score
        worst_subboard = min(sub_board_scores, key=lambda x: x[1])
    else:
        # Normalize SP scores between 0 and 1
        min_sp, max_sp = min(sp_scores), max(sp_scores)
        min_cs, max_cs = min(cs_scores), max(cs_scores)

        def normalize(value, min_val, max_val):
            return (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5

        # Normalize SP and CS, then find the worst combined sub-board
        normalized_scores = [
            (normalize(sp, min_sp, max_sp), normalize(cs, min_cs, max_cs), coords)
            for sp, cs, coords in sub_board_scores
        ]

        worst_subboard = min(normalized_scores, key=lambda x: x[0] + x[1])  # Select the worst based on sum of normalized scores

    return worst_subboard[0], worst_subboard[2]


def get_index_of_the_best_fitted_individuals(population_scores, num_individuals):
    """
    Identifies the best-fitted individuals based on the chosen evaluation mode.

    Selection is made by sorting individuals according to:
    - 'sp': Highest Sum-of-Pairs score.
    - 'cs': Highest Column Score.
    - 'mo': Normalized combination of SP and CS.

    Args:
        population_scores (list of tuples): The evaluated population scores.
        num_individuals (int): Number of top individuals to select.

    Returns:
        list: A list of indices corresponding to the best-fitted individuals.
    """
    if len(population_scores[0]) == 2:
        # If only one metric is used (SP or CS), sort directly
        sorted_population = sorted(population_scores, key=lambda x: x[1], reverse=True)
    else:
        # Normalize both SP and CS scores for fair comparison
        sp_scores = [x[1] for x in population_scores]
        cs_scores = [x[2] for x in population_scores]

        min_sp, max_sp = min(sp_scores), max(sp_scores)
        min_cs, max_cs = min(cs_scores), max(cs_scores)

        def normalize(value, min_val, max_val):
            return (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5

        # Sort based on the normalized sum of SP and CS
        normalized_population = sorted(
            [(idx, normalize(sp, min_sp, max_sp), normalize(cs, min_cs, max_cs))
             for idx, sp, cs in population_scores],
            key=lambda x: x[1] + x[2], reverse=True
        )

        sorted_population = [(idx, sp + cs) for idx, sp, cs in normalized_population]

    return [ind for ind, _ in sorted_population[:num_individuals]]


def check_if_there_are_all_gaps(row, from_index):
    """
    Check if all elements from `from_index` to the end of the row are gaps (5).

    If all elements are gaps, return the `from_index` as the position where
    gaps start. Otherwise, return False.

    Parameters:
    -----------
    - row (list): The sequence row (a list of integers representing nucleotides or gaps).
    - from_index (int): The starting index for checking gaps.

    Returns:
    --------
    - int: The index where gaps start if all elements from `from_index` onward are gaps.
    - bool: False if there is any non-gap element in the range.

    Example:
    --------
    >>> check_if_there_are_all_gaps([1, 2, 5, 5, 5], 2)
    2  # Gaps start at index 2

    >>> check_if_there_are_all_gaps([1, 2, 5, 4, 5], 2)
    False  # A non-gap element is found
    """
    for i in range(from_index, len(row)):
        if row[i] != 5:
            return False  # Found a non-gap character

    return from_index  # Return the index where gaps start


def clean_unnecessary_gaps(aligned_sequence):
    """
    Removes trailing columns that contain only gaps (5) in an MSA alignment.
    Also removes any columns where every sequence consists only of gaps.

    Parameters:
    -----------
    - aligned_sequence (list of lists): The MSA alignment matrix, where each row
                                        represents a sequence.

    Returns:
    --------
    - None: The function modifies `aligned_sequence` in place.
    """
    if not aligned_sequence or not aligned_sequence[0]:
        return  # Do nothing if sequence is empty

    # Helper: compute the index where trailing gaps begin for a given row.
    def trailing_gap_index(row):
        i = len(row)
        while i > 0 and row[i - 1] == 5:
            i -= 1
        return i

    # Step 1: Remove trailing gap columns that are gaps in every row.
    # Compute, for each row, the first index from the right where non-gaps appear.
    # The common trailing gap region (present in every row) starts at the maximum index.
    common_trailing_index = max(trailing_gap_index(row) for row in aligned_sequence)
    for row in aligned_sequence:
        del row[common_trailing_index:]  # Remove columns from common_trailing_index onward

    # Step 2: Remove any remaining columns that are entirely gaps.
    num_columns = len(aligned_sequence[0])
    # Identify columns where every row has a gap (value 5)
    all_gap_columns = [col for col in range(num_columns) if all(row[col] == 5 for row in aligned_sequence)]
    # Remove these columns in reverse order to avoid shifting indices
    for col in sorted(all_gap_columns, reverse=True):
        for row in aligned_sequence:
            del row[col]


def get_nucleotides_seqs(chromosome):
    """
    Converts a chromosome representation (numerical encoding) into nucleotide sequences.

    Each sequence in the chromosome is represented as a list of integers, where:
    - 1 -> 'A' (Adenine)
    - 2 -> 'T' (Thymine)
    - 3 -> 'C' (Cytosine)
    - 4 -> 'G' (Guanine)
    - 5 -> '-' (Gap)

    This function maps each integer back to its corresponding nucleotide character.

    Parameters:
    -----------
        chromosome (list of lists): A list of sequences, where each sequence is a list of integers.

    Returns:
    --------
        list: A list of strings, where each string represents a nucleotide sequence.

    Example:
        >>> chromosome = [[1, 2, 3, 5], [4, 3, 2, 1]]
        >>> get_nucleotides_seqs(chromosome)
        ['ATC-', 'GCAT']
    """
    # Define the nucleotide mapping for integer values
    nucleotides = ['A', 'T', 'C', 'G', '-', 'N']

    # Initialize a list to store the converted nucleotide sequences
    nucleotides_seqs = []

    # Iterate over each sequence in the chromosome
    for sequence in chromosome:
        # Convert each integer in the sequence to its corresponding nucleotide
        nucleotide_sequence = ''.join([nucleotides[n - 1] for n in sequence])
        nucleotides_seqs.append(nucleotide_sequence)

    return nucleotides_seqs


# ===========================
# Benchmarking Utilities
# ===========================
def calculate_metrics(env):
    """
    Compute key evaluation metrics for a Multiple Sequence Alignment (MSA).

    This function extracts alignment statistics from the given environment (`env`)
    and calculates various alignment quality measures.

    Parameters:
    -----------
    - env (Environment): An instance of the MSA environment, containing aligned sequences.

    Returns:
    --------
    - dict: A dictionary containing the following metrics:
        - "AL" (int): Alignment Length (number of columns in the aligned sequences).
        - "QTY" (int): Number of sequences in the alignment.
        - "SP" (float): Sum-of-Pairs (SP) score, measuring sequence similarity.
        - "EM" (int): Number of fully matched columns (Exact Matches).
        - "CS" (float): Column Score (fraction of exact match columns).

    Example:
    --------
    >>> env = Environment(["ATCG", "AT-G", "ATGG"])  # Example MSA environment
    >>> calculate_metrics(env)
    {'AL': 4, 'QTY': 3, 'SP': 12, 'EM': 2, 'CS': 0.5}
    """
    alignment_length = len(env.aligned[0])  # Number of columns in the alignment
    num_sequences = len(env.aligned)  # Total number of sequences
    sp_score = env.calc_score()  # Sum-of-Pairs score
    exact_matches = env.calc_exact_matched()  # Number of fully matched columns
    column_score = exact_matches / alignment_length  # Fraction of exact match columns

    return {
        "AL": alignment_length,
        "QTY": num_sequences,
        "SP": sp_score,
        "EM": exact_matches,
        "CS": column_score
    }


def parse_fasta_to_sequences(fasta_content):
    """
    Extract sequences from a FASTA-formatted string.

    This function processes a FASTA file/string, extracting and concatenating sequences
    while ignoring headers (lines starting with '>').

    Parameters:
    -----------
    - fasta_content (str): The content of a FASTA file as a single string.

    Returns:
    --------
    - list of str: A list of extracted sequences, where each sequence is a continuous string.

    Example:
    --------
    >>> fasta_data =
    >seq1
    ATCGGCTA
    >seq2
    TTAGCCCTA

    >>> parse_fasta_to_sequences(fasta_data)
    ['ATCGGCTA', 'TTAGCCCTA']
    """
    sequences = []  # Stores extracted sequences
    current_sequence = []  # Remove leading/trailing whitespace

    # Process each line in the FASTA file
    for line in fasta_content.splitlines():
        line = line.strip()  # Remove leading/trailing whitespace

        if line.startswith(">"):  # Sequence identifier line
            if current_sequence:
                sequences.append(''.join(current_sequence))  # Save previous sequence
                current_sequence = []  # Reset for new sequence
        else:
            current_sequence.append(line)  # Collect sequence lines

    # Add the last sequence if it exists
    if current_sequence:
        sequences.append(''.join(current_sequence))

    return sequences


def display_menu():
    """
    Display a selection menu for benchmarking different MSA methods.

    This function prompts the user to select a benchmarking comparison, ensuring
    valid input (1, 2, or 3) before returning the choice.

    Options:
    --------
    1. GA-DPAMSA vs DPAMSA
    2. GA-DPAMSA vs Other MSA Tools
    3. GA-DPAMSA vs DPAMSA vs Other MSA Tools

    Returns:
    --------
    - int: The user's selected option (1, 2, or 3).

    Example:
    --------
    >>> choice = display_menu()
    Select the benchmarking option:
    1. GA-DPAMSA vs DPAMSA
    2. GA-DPAMSA vs Other MSA Tools
    3. GA-DPAMSA vs DPAMSA vs Other MSA Tools
    Enter your choice (1, 2, or 3): 2
    >>> print(choice)
    2  # User selected option 2
    """
    print("Select the benchmarking option:")
    print("1. GA-DPAMSA vs DPAMSA")
    print("2. GA-DPAMSA vs Other MSA Tools")
    print("3. GA-DPAMSA vs DPAMSA vs Other MSA Tools")

    while True:
        try:
            # Request user input and convert to an integer
            choice = int(input("Enter your choice (1, 2, or 3): "))

            # Validate input (must be 1, 2, or 3)
            if choice in [1, 2, 3]:
                return choice
            else:
                print("Please select a valid option (1, 2, or 3).")
        except ValueError:
            print("Invalid input. Please enter a number.")


def run_tool_and_generate_report(tool_name, file_paths, dataset_name):
    """
    Run an external MSA tool, process its alignment output, and generate a benchmarking report.

    This function executes the specified MSA tool on a set of FASTA files, extracts the
    aligned sequences, computes evaluation metrics, and generates a report.

    Parameters:
    -----------
    - tool_name (str): The name of the MSA tool (must be in `config.TOOLS`).
    - file_paths (list of str): List of paths to input FASTA files.
    - dataset_name (str): Name of the dataset (used for organizing output files).

    Returns:
    --------
    - list of lists: A list containing alignment evaluation metrics for each processed file.

    Each entry is a list with: [file_name, AL (Alignment Length), QTY (Number of Sequences), SP (Sum of Pairs Score), EM (Exact Matches), CS (Column Score)].
    """
    tool_info = config.TOOLS[tool_name]

    # Create necessary directories for output and reports
    os.makedirs(tool_info['output_dir'], exist_ok=True)
    dataset_output_dir = os.path.join(tool_info['output_dir'], dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    os.makedirs(tool_info['report_dir'], exist_ok=True)

    report_file = os.path.join(tool_info['report_dir'], f"{dataset_name}.txt")
    csv_results = []

    with open(report_file, 'w') as report:
        for file_path in tqdm(file_paths, desc=f"Processing {tool_name}", leave=False):
            file_name = os.path.basename(file_path)
            file_name_no_ext = os.path.splitext(file_name)[0]
            command = tool_info['command'](file_path, os.path.join(dataset_output_dir, file_name))

            # Execute the tool's command
            if tool_name == 'MAFFT':  # Tool specific execution
                subprocess.run(command, shell=True, stderr=subprocess.DEVNULL, text=True)
            else:
                subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

            # Handle specific output paths for UPP and PASTA
            if tool_name == 'UPP':
                output_path = os.path.join(dataset_output_dir, file_name, "output_alignment.fasta")
            elif tool_name == 'PASTA':
                output_path = os.path.join(dataset_output_dir, file_name, f"pastajob.marker001.{file_name_no_ext}.aln")
            else:
                output_path = os.path.join(dataset_output_dir, file_name)

            # Read the tool's output (either from file or stdout)
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    fasta_content = f.read()
            else:
                fasta_content = subprocess.run(command, stdout=subprocess.PIPE, text=True).stdout

            if not fasta_content.strip():
              print(f"[WARN] Empty output from {tool_name} for the file {file_name}.")
              continue
          
            # Parse FASTA content to get aligned sequences
            aligned_seqs = parse_fasta_to_sequences(fasta_content)

            if not aligned_seqs:
              print(f"[WARN] parse_fasta_to_sequences could not find any sequence for {file_name} "
                    f"(tool={tool_name}). Check output format.")
              continue
                
            # Compute alignment metrics using Environment
            env = Environment(aligned_seqs, convert_data=False)
            Environment.set_alignment(env, aligned_seqs)

            metrics = calculate_metrics(env)

            # Write metrics to report
            report.write(f"File: {file_name}\n")
            report.write(f"Number of Sequences (QTY): {metrics['QTY']}\n")
            report.write(f"Alignment Length (AL): {metrics['AL']}\n")
            report.write(f"Sum of Pairs (SP): {metrics['SP']}\n")
            report.write(f"Exact Matches (EM): {metrics['EM']}\n")
            report.write(f"Column Score (CS): {metrics['CS']:.3f}\n")
            report.write(f"Alignment:\n{env.get_alignment()}\n\n")

            # Store results for CSV export
            csv_results.append([
                file_name, metrics['QTY'], metrics['AL'],
                metrics['SP'], metrics['EM'], metrics['CS']
            ])

            # Remove ClustalW .dnd files (temporary files used during alignment)
            if tool_name == 'ClustalW':
                dnd_files = glob.glob(os.path.join(os.path.dirname(file_path), '*.dnd'))
                for dnd_file in dnd_files:
                    os.remove(dnd_file)

    return csv_results


def save_inference_csv(csv_data, tool_name, dataset_name):
    """
    Save inference results to a CSV file for later analysis.

    This function stores benchmarking results, ensuring that alignment evaluation
    metrics are saved for different MSA tools.

    Parameters:
    -----------
    - csv_data (list of lists or str): If a list, it contains the alignment evaluation
                                       metrics. If a string, it represents the path
                                       to an existing CSV file.
    - tool_name (str): The name of the MSA tool (used to organize results).
    - dataset_name (str): The dataset name (used for naming the output file).

    Returns:
    --------
    - str: The file path of the saved CSV file.

    Example:
    --------
    >>> csv_data = [
    ...     ["dataset1.fasta", 150, 5, 120, 50, 0.65],
    ...     ["dataset2.fasta", 140, 4, 110, 45, 0.60]
    ... ]
    >>> save_inference_csv(csv_data, "ClustalW", "Dataset1")
    'path/to/csv/ClustalW/Dataset1_ClustalW_results.csv'
    """
    # Directory where the CSV will be stored
    tool_csv_dir = os.path.join(config.CSV_PATH, tool_name)
    os.makedirs(tool_csv_dir, exist_ok=True)

    # Define CSV file name
    csv_file_path = os.path.join(tool_csv_dir, f"{dataset_name}_{tool_name}_results.csv")

    # Convert input data to DataFrame if necessary
    if isinstance(csv_data, list):
        columns = ["File Name", "Alignment Length (AL)",
                   "Number of Sequences (QTY)", "Sum of Pairs (SP)",
                   "Exact Matches (EM)", "Column Score (CS)"]
        df = pd.DataFrame(csv_data, columns=columns)
    else:
        # If csv_data is a CSV file path, load it as a DataFrame
        df = pd.read_csv(csv_data)

    # Save DataFrame to CSV
    df.to_csv(csv_file_path, index=False)

    return csv_file_path  # Return the file path for tracking

from dataset_module import FastaDataset

def run_ga_dpamsa_inference(mode, dataset:FastaDataset, model_path):
    """
    Run inference using GA-DPAMSA and return the results CSV file path.

    This function executes GA-DPAMSA (Genetic Algorithm-enhanced DPAMSA) on a given
    dataset using a specified trained model, then returns the path to the CSV file
    containing alignment evaluation metrics.

    Parameters:
    -----------
    - mode (str): The mode of operation. Must be one of:
            * 'sp'  -> Sum of Pairs mode
            * 'cs'  -> Column Score mode
            * 'mo'  -> Multi-Objective mode
    - dataset (module): The dataset object that represents the module containing 
                        sequences to be aligned.
    - model_path (str): Path to the trained GA-DPAMSA model.

    Returns:
    --------
    - str: Path to the CSV file where inference results are saved.
    """
    from mainGA import inference as ga_inference

    # Run GA-DPAMSA inference
    ga_inference(mode=mode, dataset=dataset, model_path=model_path)

    # Construct and return the CSV results file path
    mode_tag = {"sp": "Max_SP", "cs": "Max_CS", "mo": "MO"}[mode]
    return os.path.join(config.GA_DPAMSA_INF_CSV_PATH, f"{dataset.name}_{mode_tag}_GA_DPAMSA_results.csv")


def run_dpamsa_inference(dataset: FastaDataset, model_path):
    """
    Run inference using DPAMSA and return the results CSV file path.

    This function executes DPAMSA (Deep reinforcement learning-based MSA) on a given
    dataset using a specified trained model, then returns the path to the CSV file
    containing alignment evaluation metrics.

    Parameters:
    -----------
    - dataset (module): The dataset module containing sequences to be aligned.
    - dataset_name (str): The name of the dataset (used for naming output files).
    - model_path (str): Path to the trained DPAMSA model.

    Returns:
    --------
    - str: Path to the CSV file where inference results are saved.
    """
    from DPAMSA.main import inference as dpamsa_inference

    # Run DPAMSA inference
    dpamsa_inference(dataset=dataset, model_path=model_path, truncate_file=True)

    # Construct and return the CSV results file path
    return os.path.join(config.DPAMSA_INF_CSV_PATH, f"{dataset.name}_DPAMSA_results.csv")


# ===========================
# Data Visualization Utilities
# ===========================
def plot_metrics(tool_csv_paths, dataset_name):
    """
    Generate visualizations comparing MSA tool performance.

    This function reads CSV result files for different MSA tools, extracts evaluation metrics,
    and generates box plots (for distribution) and bar plots (for mean values) of:
    - Sum of Pairs (SP) score
    - Column Score (CS)

    Parameters:
    -----------
    - tool_csv_paths (dict): Dictionary mapping tool names to their result CSV file paths.
    - dataset_name (str): Name of the dataset (used for organizing output charts).
    """
    sum_of_pairs_data = []
    column_score_data = []
    mean_sp = {}
    mean_cs = {}

    # Define colors: Red for GA-DPAMSA, Cyan for other tools
    color_map = {'GA-DPAMSA': 'red'}

    # Create output directory for dataset charts
    dataset_charts_dir = os.path.join(config.CHARTS_PATH, dataset_name)
    os.makedirs(dataset_charts_dir, exist_ok=True)

    # Process CSV files and extract metrics
    for tool, csv_path in tool_csv_paths.items():
        df = pd.read_csv(csv_path)

        # Assign colors to tools
        color = 'red' if tool == 'GA-DPAMSA' else 'turquoise'
        color_map[tool] = color

        # Store box plot data
        sum_of_pairs_data.append((tool, df['Sum of Pairs (SP)']))
        column_score_data.append((tool, df['Column Score (CS)']))

        # Compute mean values for bar plots
        mean_sp[tool] = df['Sum of Pairs (SP)'].mean()
        mean_cs[tool] = df['Column Score (CS)'].mean()

    tools = list(tool_csv_paths.keys())

    # === BOX PLOT: Sum of Pairs (SP) ===
    plt.figure(figsize=(12, 8))
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    boxplot = plt.boxplot(
        [data for _, data in sum_of_pairs_data],
        labels=tools,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1),
        zorder=3
    )

    # Apply colors to box plots
    for patch, (tool, _) in zip(boxplot['boxes'], sum_of_pairs_data):
        patch.set_facecolor(color_map[tool])

    # Aesthetics
    plt.title(f'SP Distribution results for {dataset_name}', fontsize=16, fontweight='bold')
    plt.ylabel('Sum of Pairs (SP)', fontweight='bold', fontsize=12)
    plt.xticks(fontweight='bold', fontsize=10)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(dataset_charts_dir, f'sum_of_pairs_distribution.png'), dpi=300)
    plt.close()

    # === BOX PLOT: Column Score (CS) ===
    plt.figure(figsize=(12, 8))
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    boxplot = plt.boxplot(
        [data for _, data in column_score_data],
        labels=tools,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1),
        zorder=3
    )

    # Apply colors to box plots
    for patch, (tool, _) in zip(boxplot['boxes'], column_score_data):
        patch.set_facecolor(color_map[tool])

    # Aesthetics
    plt.title(f'CS Distribution results for {dataset_name}', fontsize=16, fontweight='bold')
    plt.ylabel('Column Score (CS)', fontweight='bold', fontsize=12)
    plt.xticks(fontweight='bold', fontsize=10)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(dataset_charts_dir, f'column_score_distribution.png'), dpi=300)
    plt.close()

    # === BAR PLOT: Mean Sum of Pairs (SP) ===
    plt.figure(figsize=(12, 8))
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    bars = plt.bar(
        mean_sp.keys(),
        mean_sp.values(),
        color=[color_map[tool] for tool in mean_sp.keys()],
        edgecolor='black', linewidth=1.2,
        zorder=3
    )

    # Add explicit mean value to bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.1,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=12
        )

    plt.title(f'Mean SP results for {dataset_name}', fontsize=16, fontweight='bold')
    plt.ylabel('Mean SP', fontweight='bold', fontsize=12)
    plt.xticks(fontweight='bold', fontsize=10)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(dataset_charts_dir, f'mean_sum_of_pairs.png'), dpi=300)
    plt.close()

    # === BAR PLOT: Mean Column Score (CS) ===
    plt.figure(figsize=(12, 8))
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    bars = plt.bar(
        mean_cs.keys(),
        mean_cs.values(),
        color=[color_map[tool] for tool in mean_cs.keys()],
        edgecolor='black', linewidth=1.2,
        zorder=3
    )

    # Add explicit mean value to bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + (height * 0.015),
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=12
        )

    plt.title(f'Mean CS results for {dataset_name}', fontsize=16, fontweight='bold')
    plt.ylabel('Mean CS', fontweight='bold', fontsize=12)
    plt.xticks(fontweight='bold', fontsize=10)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(dataset_charts_dir, f'mean_column_score.png'), dpi=300)
    plt.close()
