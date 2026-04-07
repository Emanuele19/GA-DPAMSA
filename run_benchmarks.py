import os
from tqdm import tqdm

import config
import utils

from dataset_module import FastaDataset, SequenceEncoder


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

DATASET_NAME = 'orthodb_v12/inference_benchmark_ready'  # Name of the dataset to benchmark (must match generated dataset)
DPAMSA_MODEL = 'model_3x30'
GA_DPAMSA_MODEL = 'model_3x30'
DCNNMSA_MODEL = 'msa_model_ep18999.pth'
PPO_MODEL ='final_model_PPO.pt'
GRPO_MODEL ='final_model_GRPO.pt'

encoder = SequenceEncoder(config.NUCLEOTIDE_ENCODING)

# ===========================
# Main Function
# ===========================


import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


def _run_external_tool(tool_name, file_paths, dataset_name):
    """
    Helper function executed in a separate process for each external tool.
    Returns (tool_name, csv_path).
    """
    tool_results = utils.run_tool_and_generate_report(tool_name, file_paths, dataset_name)
    csv_path = utils.save_inference_csv(tool_results, tool_name, dataset_name)
    return tool_name, csv_path

def _run_nsga_worker(dataset_path, model_name):
    print("Running NSGA-II...")
    dataset = FastaDataset(dataset_path)
    csv_path = utils.run_nsga_inference(dataset, model_name)
    print("NSGA-II finished. CSV:", csv_path)
    return "NSGA-II", csv_path

def _run_ga_dpamsa_worker(dataset_path, model_name):
    """
    Worker process for GA-DPAMSA.
    """
    dataset = FastaDataset(dataset_path, encoder=encoder)
    csv_path = utils.run_ga_dpamsa_inference('sp', dataset, model_name)
    return "GA-DPAMSA", csv_path


def _run_dpamsa_worker(dataset_path, model_name):
    """
    Worker process for DPAMSA.
    """
    dataset = FastaDataset(dataset_path, encoder=encoder)
    csv_path = utils.run_dpamsa_inference(dataset, model_name)
    return "DPAMSA", csv_path


def _run_dcnnmsa_worker(dataset_path, model_name):
    """
    Worker process for DCNNMSA.
    """
    dataset = FastaDataset(dataset_path, encoder=encoder)
    from DCNN_BDDQN.inference import run_inference

    csv_path = os.path.join(config.INFERENCE_CSV_PATH, 'DCNNMSA/DCNN_BDDQN_results.csv')
    out_path = os.path.join(config.REPORTS_PATH, 'DCNNMSA/DCNN_BDDQN_results.txt')
    run_inference(
        model_path = os.path.join(config.MODEL_WEIGHTS_PATH, DCNNMSA_MODEL),
        data_folder = os.path.join(config.FASTA_FILES_PATH, DATASET_NAME),
        csv_file = csv_path,
        output_file = out_path
    )
    return "DCNNMSA", csv_path

def _run_oneshot_rl_worker(dataset_path, model_name, algo_name):
    """
    Worker process per PPO e GRPO.
    algo_name sarà "PPO" o "GRPO" e serve a salvare i file nelle cartelle giuste.
    """
    from ppo_grpo.inference import run_inference
    
    csv_path = os.path.join(config.INFERENCE_CSV_PATH, f'{algo_name}/{algo_name}_results.csv')
    out_path = os.path.join(config.REPORTS_PATH, f'{algo_name}/{algo_name}_results.txt')
    
    run_inference(
        model_path = os.path.join(config.MODEL_WEIGHTS_PATH, model_name),
        data_folder = dataset_path,
        csv_file = csv_path,
        output_file = out_path,
        algo_name = algo_name
    )
    return algo_name, csv_path

# Worker function per NSGA-II + DCNN_BDDQN
def _run_nsga_dcnn_worker(dataset_path, model_name):
    print("Running NSGA-II + DCNN_BDDQN...")
    dataset = FastaDataset(dataset_path, encoder=encoder)
    
    # Richiama una funzione dedicata in utils per gestire questo specifico GA
    csv_path = utils.run_nsga_dcnn_inference(dataset, model_name)
    
    print("NSGA-II + DCNN_BDDQN finished. CSV:", csv_path)
    return "NSGA-DCNN", csv_path

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

    # Creiamo un contesto 'spawn' sicuro per le GPU
    ctx = mp.get_context('spawn')
    
    # Passiamo il contesto all'executor
    with ProcessPoolExecutor(mp_context=ctx) as executor:
        # GA-DPAMSA
        jobs.append(
            executor.submit(_run_ga_dpamsa_worker, dataset_path, GA_DPAMSA_MODEL)
        )

        # new GA NSGA
        jobs.append(
            executor.submit(_run_nsga_worker, dataset_path, GA_DPAMSA_MODEL)
            )
        

        jobs.append(
            executor.submit(_run_dcnnmsa_worker, dataset_path, DCNNMSA_MODEL)
        )

        jobs.append(executor.submit(_run_nsga_dcnn_worker, dataset_path, DCNNMSA_MODEL))

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
        
        if choice == 4 or choice == 3:
            jobs.append(
                executor.submit(_run_oneshot_rl_worker, dataset_path, GRPO_MODEL, "GRPO")
            )
            jobs.append(
                executor.submit(_run_oneshot_rl_worker, dataset_path, PPO_MODEL, "PPO")
            )

        # Progress tracking
        for future in tqdm(as_completed(jobs), total=len(jobs), desc="Running benchmarks"):
            tool_name, csv_path = future.result()
            tool_csv_paths[tool_name] = csv_path

    # Generate performance plots for the selected tools
    utils.plot_metrics(tool_csv_paths, DATASET_NAME)

    # Generate compact benchmark report comparing tools
    utils.generate_compact_benchmark_report(tool_csv_paths, DATASET_NAME, dataset_path)


if __name__ == "__main__":
    main()
