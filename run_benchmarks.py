import os
from tqdm import tqdm

import config
import utils

from dataset_module import FastaDataset

"""
Benchmarking Script for MSA Methods

This script benchmarks different Multiple Sequence Alignment (MSA) methods, including:
- GA-DPAMSA (Genetic Algorithm-enhanced DPAMSA)
- DPAMSA (Deep Reinforcement Learning-based MSA)
- Other external MSA tools (ClustalW, MAFFT, MUSCLE, etc.)

It allows the user to select benchmarking options, executes the selected MSA methods, 
and generates reports and performance visualizations.

Author: https://github.com/FLaTNNBio/GA-DPAMSA
"""

# ===========================
# Dataset and Model Configuration
# ===========================

# Ensure the dataset name matches the imported dataset module
DATASET_NAME = 'dataset1_3x30bp'

# Ensure DPAMSA model matches dataset size
DPAMSA_MODEL = 'model_3x30'

# Ensure GA-DPAMSA model matches 'AGENT_WINDOW_ROW' and 'AGENT_WINDOW_COLUMN' settings
GA_DPAMSA_MODEL = 'model_3x30'


def main():
    """
    Main function to execute MSA benchmarking.

    - Displays a selection menu for benchmarking options.
    - Runs GA-DPAMSA inference (always executed).
    - Runs DPAMSA inference if selected.
    - Runs external MSA tools if selected.
    - Saves results and generates performance plots.
    """
    # Display selection menu
    choice = utils.display_menu()

    # Get dataset file paths
    dataset_folder = os.path.join(config.FASTA_FILES_PATH, DATASET_NAME)
    file_paths = [os.path.join(dataset_folder, file) for file in sorted(os.listdir(dataset_folder))]

    # Dictionary to store CSV paths for each tool
    tool_csv_paths = {}

    dataset = FastaDataset(os.path.join(config.FASTA_FILES_PATH, DATASET_NAME))

    # GA-DPAMSA must always be executed
    ga_csv_path = utils.run_ga_dpamsa_inference('sp', dataset, GA_DPAMSA_MODEL)
    tool_csv_paths['GA-DPAMSA'] = ga_csv_path

    # Run DPAMSA if option 1 or 3 is selected
    if choice == 1 or choice == 3:
        dpamsa_csv_path = utils.run_dpamsa_inference(dataset, DPAMSA_MODEL)
        tool_csv_paths['DPAMSA'] = dpamsa_csv_path

    # Run external MSA tools if option 2 or 3 is selected
    if choice == 2 or choice == 3:
        for tool_name in tqdm(config.TOOLS.keys(), desc="Processing Tools"):
            tool_results = utils.run_tool_and_generate_report(tool_name, file_paths, DATASET_NAME)
            csv_path = utils.save_inference_csv(tool_results, tool_name, DATASET_NAME)  # Save CSV and get path
            tool_csv_paths[tool_name] = csv_path

    # Generate performance plots for the selected tools
    utils.plot_metrics(tool_csv_paths, DATASET_NAME)


if __name__ == "__main__":
    main()
