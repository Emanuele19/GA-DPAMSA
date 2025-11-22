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

from concurrent.futures import ProcessPoolExecutor, as_completed


def _run_external_tool(tool_name, file_paths, dataset_name):
    """
    Helper function executed in a separate process for each external tool.
    Returns (tool_name, csv_path).
    """
    tool_results = utils.run_tool_and_generate_report(tool_name, file_paths, dataset_name)
    csv_path = utils.save_inference_csv(tool_results, tool_name, dataset_name)
    return tool_name, csv_path


def _run_ga_dpamsa_worker(dataset_path, model_name):
    """
    Worker process for GA-DPAMSA.
    """
    dataset = FastaDataset(dataset_path)
    csv_path = utils.run_ga_dpamsa_inference('sp', dataset, model_name)
    return "GA-DPAMSA", csv_path


def _run_dpamsa_worker(dataset_path, model_name):
    """
    Worker process for DPAMSA.
    """
    dataset = FastaDataset(dataset_path)
    csv_path = utils.run_dpamsa_inference(dataset, model_name)
    return "DPAMSA", csv_path


def main():
    """
    Main function to execute MSA benchmarking.

    - Displays a selection menu for benchmarking options.
    - Runs GA-DPAMSA inference (always executed).
    - Runs DPAMSA inference if selected.
    - Runs external MSA tools if selected (in parallelo).
    - Saves results and generates performance plots.
    """
    # Display selection menu
    choice = utils.display_menu()

    # Paths
    dataset_folder = os.path.join(config.FASTA_FILES_PATH, DATASET_NAME)
    dataset_path = os.path.join(config.FASTA_FILES_PATH, DATASET_NAME)
    file_paths = [os.path.join(dataset_folder, file) for file in sorted(os.listdir(dataset_folder))]

    # Dictionary to store CSV paths for each tool
    tool_csv_paths = {}

    # Costruiamo la lista di job da lanciare in parallelo
    jobs = []

    with ProcessPoolExecutor() as executor:
        # GA-DPAMSA
        jobs.append(
            executor.submit(_run_ga_dpamsa_worker, dataset_path, GA_DPAMSA_MODEL)
        )

        # DPAMSA if choice is 1 or 3
        if choice == 1 or choice == 3:
            jobs.append(
                executor.submit(_run_dpamsa_worker, dataset_path, DPAMSA_MODEL)
            )

        # External tools for choice 2 or 3
        if choice == 2 or choice == 3:
            tools = list(config.TOOLS.keys())
            for tool_name in tools:
                jobs.append(
                    executor.submit(_run_external_tool, tool_name, file_paths, DATASET_NAME)
                )

        # Progress tracking
        for future in tqdm(as_completed(jobs), total=len(jobs), desc="Running benchmarks"):
            tool_name, csv_path = future.result()
            tool_csv_paths[tool_name] = csv_path

    # Generate performance plots for the selected tools
    utils.plot_metrics(tool_csv_paths, DATASET_NAME)


if __name__ == "__main__":
    main()
