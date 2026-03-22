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

BASE_DS_NAME = 'hdf5_3x30_test'
DATASET_NAME = f'orthodb_v12/{BASE_DS_NAME}'
DPAMSA_MODEL = 'model_3x30'
GA_DPAMSA_MODEL = 'model_3x30'
DCNNMSA_MODEL = 'msa_model_ep18999.pth'
DCNN_BDDQNMSA_MODEL = 'msa_model_ep18999.pth'


encoder = SequenceEncoder(config.NUCLEOTIDE_ENCODING)

# ===========================
# Main Function
# ===========================


from concurrent.futures import ProcessPoolExecutor, as_completed


def _run_external_tool(tool_name, file_paths, dataset_name):
    """
    Helper function executed in a separate process for each external tool.
    Returns (tool_name, csv_path).
    """
    # This function now returns a list of dictionaries directly
    tool_results = utils.run_tool_and_get_metrics(tool_name, file_paths, dataset_name)
    
    if not tool_results:
        print(f"Warning: No results returned from {tool_name} for dataset {dataset_name}. Skipping file save.")
        # Return a placeholder or handle it as an error
        return tool_name, None

    # Define paths for report and CSV
    report_path = os.path.join(config.TOOLS[tool_name]['report_dir'], f"{dataset_name}.txt")
    csv_path = os.path.join(config.TOOLS[tool_name]['report_dir'], f"{dataset_name}_{tool_name}_results.csv")

    # Ensure directories exist
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Save the results to disk (both .txt report and .csv)
    utils.save_to_disk(tool_results, report_path, csv_path)
    
    return tool_name, csv_path


def _run_model_worker(model_name, dataset_path, inference_function, inference_params=None):
    """
    Generic worker process for a model.
    """
    if inference_params is None:
        inference_params = {}

    model_config = config.MODELS[model_name]
    model_path = model_config["model_path"]

    if model_config["requires_dataset_object"]:
        dataset = FastaDataset(dataset_path, encoder=encoder)
        results = inference_function(dataset=dataset, model_path=model_path, **inference_params)
    else:
        results = inference_function(data_folder=dataset_path, model_path=model_path, **inference_params)

    # Define paths for report and CSV using config
    base_dataset_name = os.path.basename(dataset_path)
    
    if model_name == "GA-DPAMSA":
        mode_tag = {"sp": "Max_SP", "cs": "Max_CS", "mo": "MO"}[inference_params.get("mode", "sp")]
        report_path = os.path.join(model_config["report_path"], f"{base_dataset_name}_{mode_tag}.txt")
        csv_path = os.path.join(model_config["csv_path"], f"{base_dataset_name}_{mode_tag}_GA_DPAMSA_results.csv")
    else:
        report_path = os.path.join(model_config["report_path"], f"{base_dataset_name}_{model_name}_results.txt")
        csv_path = os.path.join(model_config["csv_path"], f"{base_dataset_name}_{model_name}_results.csv")


    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    utils.save_to_disk(results, report_path, csv_path)
    
    return model_name, csv_path


def main():
    """
    Main function to execute MSA benchmarking.
    """
    choice = utils.display_menu()

    dataset_folder = os.path.join(config.FASTA_FILES_PATH, DATASET_NAME)
    file_paths = [os.path.join(dataset_folder, f) for f in sorted(os.listdir(dataset_folder))]

    tool_csv_paths = {}
    jobs = []

    # Map model names to their inference functions
    from DCNNMSA.inference import run_inference as dcnn_inference
    from DCNN_BDDQNMSA.inference import run_inference as dcnn_bddqn_inference
    
    inference_functions = {
        "GA-DPAMSA": utils.run_ga_dpamsa_inference,
        "DPAMSA": utils.run_dpamsa_inference,
        "DCNNMSA": dcnn_inference,
        "DCNN_BDDQNMSA": dcnn_bddqn_inference,
    }

    # Define which models to run based on user choice
    models_to_run = ["GA-DPAMSA", "DCNNMSA", "DCNN_BDDQNMSA"]
    if choice == 1 or choice == 3:
        models_to_run.append("DPAMSA")

    with ProcessPoolExecutor() as executor:
        # Submit model benchmarks
        for model_name in models_to_run:
            inference_params = {"mode": "sp"} if model_name == "GA-DPAMSA" else {}
            jobs.append(
                executor.submit(
                    _run_model_worker,
                    model_name,
                    dataset_folder,
                    inference_functions[model_name],
                    inference_params
                )
            )

        # Submit external tool benchmarks
        if choice == 2 or choice == 3:
            for tool_name in config.TOOLS.keys():
                jobs.append(
                    executor.submit(_run_external_tool, tool_name, file_paths, BASE_DS_NAME)
                )

        # Process results
        for future in tqdm(as_completed(jobs), total=len(jobs), desc="Running benchmarks"):
            name, csv_path = future.result()
            tool_csv_paths[name] = csv_path
            print(f"[DEBUG]: {name} completed")

    # Generate performance plots
    utils.plot_metrics(tool_csv_paths, DATASET_NAME)
    print("Plotted results.")


if __name__ == "__main__":
    main()
