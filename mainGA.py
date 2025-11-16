import csv
import os
from tqdm import tqdm

import config
from DPAMSA.env import Environment
from GA import GA
import utils

"""
GA-DPAMSA Inference Script
---------------------------
This script runs the Genetic Algorithm (GA) pipeline as part of the GA-DPAMSA framework 
for Multiple Sequence Alignment (MSA). The GA is used to evolve a population of alignment 
solutions over a number of iterations. At each iteration, the pipeline performs mutation 
(using a Reinforcement Learning agent), selection, and horizontal crossover to improve the 
alignment quality. The quality is evaluated using metrics such as the Sum-of-Pairs (SP) score 
and Column Score (CS), or a combination of both in Multi-Objective (MO) mode. A Hall of Fame 
(HoF) is maintained to store the best individual (alignment) found across all generations.

Key functionalities:
  - Initialize a population of alignment solutions.
  - Evolve the population over multiple iterations using mutation, selection, and crossover.
  - Maintain a Hall of Fame of the best individual (alignment) found so far.
  - Compute and report alignment metrics (e.g., SP, CS, exact matches, alignment length).
  - Generate report and CSV files summarizing the performance for each dataset.

Usage:
  Run this script as the main entry point to perform GA-DPAMSA inference on the specified dataset. 
  The script processes the dataset(s), evolves alignments using the GA pipeline, and outputs a report 
  and CSV file with evaluation metrics.

Author: https://github.com/FLaTNNBio/GA-DPAMSA
"""

# - GA_MODE:
#       Defines the evaluation mode used by the GA. Available options are:
#         • 'sp'  → Sum-of-Pairs mode, which optimizes based on pairwise matching scores.
#         • 'cs'  → Column Score mode, which focuses on maximizing the fraction of exactly
#                   matched columns.
#         • 'mo'  → Multi-Objective mode, combining SP and CS metrics for a balanced evaluation.
#       Choose based on the alignment criteria you want to optimize.
GA_MODE = 'sp'

# Dataset module containing the sequences to be aligned.
DATASET = os.path.join(config.FASTA_FILES_PATH, 'dataset1_3x30bp')

# Identifier or path to the trained RL model used for mutation.
INFERENCE_MODEL = 'new_model_3x30'

# Debug mode flag: set to True for detailed logging, False for normal operation.
DEBUG_MODE = False


def output_parameters():
    print('\n')
    print("-------- Genetic Algorithm parameters ---------")
    print(f"Window size:{config.AGENT_WINDOW_ROW}x{config.AGENT_WINDOW_COLUMN}")
    print(f"Population Size: {config.POPULATION_SIZE}")
    print(f"Number of iteration: {config.GA_ITERATIONS}")
    print(f"Clone Rate: {config.CLONE_RATE * 100}%")
    print(f"Gap Rate: {config.GAP_RATE * 100}%")
    print(f"Selection Rate: {config.SELECTION_RATE * 100}%")
    print(f"Mutation Rate: {config.MUTATION_RATE * 100}%")
    print('\n')

from dataset_module import FastaDataset, FastaContent

def inference(
        mode, 
        dataset:FastaDataset=DATASET, 
        start=0, end=-1, 
        model_path='model_3x30', 
        debug=False, 
        truncate_file=True):
    """
        Run the genetic algorithm with a specific inference mode.

        Parameters:
        -----------
        - mode (str): The mode of operation. Must be one of:
            * 'sp'  -> Sum of Pairs mode
            * 'cs'  -> Column Score mode
            * 'mo'  -> Multi-Objective mode
        - dataset: The dataset containing sequences.
        - start (int): Starting index of datasets to process.
        - end (int): Ending index of datasets to process (default: -1 for all).
        - model_path (str): Path to the model used for mutation.
        - debug (bool): Whether to run GA in debug mode (Detailed Real-Time vision of the algorithm operating) or not.
        - truncate_file (bool): Whether to overwrite report and CSV files.

        Raises:
        -------
        - Exception: If an invalid mode is provided.
        """
    # Mode validation
    valid_modes = {'sp', 'cs', 'mo'}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Choose one of {valid_modes}.")

    # Paths definition
    mode_tag = {"sp": "Max_SP", "cs": "Max_CS", "mo": "MO"}[mode]
    report_file_name = os.path.join(config.GA_DPAMSA_REPORTS_PATH, f"{dataset.name}_{mode_tag}.txt")
    csv_file_name = os.path.join(config.GA_DPAMSA_INF_CSV_PATH, f"{dataset.name}_{mode_tag}_GA_DPAMSA_results.csv")

    # Create or truncate results files
    if truncate_file:
        with open(report_file_name, 'w'):
            pass
        with open(csv_file_name, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["File Name", "Number of Sequences (QTY)", "Alignment Length (AL)", "Sum of Pairs (SP)",
                             "Exact Matches (EM)", "Column Score (CS)"])

    # Show DPAMSA and GA configs
    output_parameters()

    # Inference loop
    datasets_to_process:list[FastaContent] = dataset[start:end if end != -1 else len(dataset)]
    for index, fasta_content in enumerate(tqdm(datasets_to_process, desc="Processing Datasets"), start):
        # Extract sequences
        seqs = fasta_content.sequences

        # Initialize Environment
        env = Environment(seqs, convert_data=False)

        # Initialize and run GA
        ga = GA(seqs, mode)
        best_alignment = ga.run(model_path, debug)

        # Set alignment to use env utilities
        Environment.set_alignment(env, best_alignment)

        # Compute metrics
        metrics = utils.calculate_metrics(env)

        # Create report
        report = (
            f"File: {fasta_content.name}\n"
            f"Number of Sequences (QTY): {metrics['QTY']}\n"
            f"Alignment Length (AL): {metrics['AL']}\n"
            f"Sum of Pairs (SP): {metrics['SP']}\n"
            f"Exact Matches (EM): {metrics['EM']}\n"
            f"Column Score (CS): {metrics['CS']:.3f}\n"
            f"Alignment:\n{env.get_alignment()}\n\n"
        )

        # Save results to files
        with open(report_file_name, 'a') as report_file:
            report_file.write(report)

        with open(csv_file_name, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([fasta_content.name, metrics['QTY'], metrics['AL'], metrics['SP'], metrics['EM'], metrics['CS']])

    print(f"\nInference completed successfully.")
    print(f"Report saved at: {report_file_name}")
    print(f"CSV saved at: {csv_file_name}\n\n")


if __name__ == "__main__":
    """
       Available inference modes:
       - 'sp'  -> Sum of Pairs mode
       - 'cs'  -> Column Score mode
       - 'mo'  -> Multi-Objective mode
    """
    default_dataset = FastaDataset()
    inference(mode=GA_MODE, dataset=DATASET, model_path=INFERENCE_MODEL, debug=DEBUG_MODE)
