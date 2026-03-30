import torch
import os
from tqdm import tqdm
from typing import List
from ppo_grpo.data.integer import IntegerStatePreprocessor
from utils import setup_logger

from ppo_grpo.env import Environment

from ppo_grpo.agent.backbone.new import RobustBackbone
from ppo_grpo.agent.output_adapter.gaussian import GaussianGapAdapter
from ppo_grpo.agent.grpo_agent import GRPO_Agent
from ppo_grpo.agent.ppo_agent import PPO_Agent
from dataset_module import FastaDataset, SequenceEncoder, SequenceDecoder
from ppo_grpo.agent.actor.mlp import MSA_Actor

import config, csv

def run_inference(model_path: str, data_folder: str, output_file: str, csv_file: str, n_seq: int = 3, vocab_size: int = 6):
    logger = setup_logger()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting Inference on device: {device}")

    # Uses Dilated CNNs to capture DNA motifs
    backbone = RobustBackbone(
        num_rows=config.AGENT_WINDOW_ROW,  # e.g., 3 sequences per block
        vocab_size=len(config.NUCLEOTIDES_MAP) + 1,  # +1 for safe padding handling
        embedding_dim=128,
        hidden_dim=128,
        num_layers=3,
    )

    # B. Output Adapter (The 'Translator')
    # Maps network output to Mean/Std for Gaussian Sampling of gaps
    adapter = GaussianGapAdapter(
        max_gaps=config.MAX_GAPS_PER_POS,
        min_log_std=-2.0,
        max_log_std=2.0,

    )
    # 1. Setup Agente e Caricamento
    # agente può essere PPO o GRPO, ma in questo caso usiamo GRPO
    actor = MSA_Actor(backbone, adapter)  # Passiamo None, caricheremo

    agent = GRPO_Agent(actor, device)

    # Carichiamo i pesi
    model_name = os.path.basename(model_path)
    model_dir = os.path.dirname(model_path)
    agent.load(os.path.join(model_dir, model_name))

    agent.eval() # Fondamentale: disattiva dropout/batchnorm

    dataset = FastaDataset(data_folder, SequenceEncoder(config.NUCLEOTIDE_ENCODING))
    preprocessor = IntegerStatePreprocessor(config, device=device)
    decoder = SequenceDecoder(config.NUCLEOTIDE_DECODING)

    inference_results = []

    print(f"Inizio Inferenza su {len(dataset)} file...")

    with torch.no_grad():
        for fasta_content in tqdm(dataset):
            # Preprocessing (Rimuoviamo gap preesistenti e passiamo a tensore)
            # FastaDataset restituisce una lista di liste di interi
            raw_seqs = fasta_content.sequences 
            state_tensor = preprocessor([raw_seqs], sanitize=True) 
            
            # Convertiamo in numpy array per darlo in pasto a get_action
            state_np = state_tensor.cpu().numpy()

            # Azione dell'Agente (One-Shot e Deterministica)
            # deterministic=True significa che prendiamo la media della Gaussiana, senza esplorazione random
            gap_matrix_batch = agent.get_action(state_np, deterministic=True)
            gap_matrix = gap_matrix_batch[0].tolist() # Estraiamo il primo (e unico) elemento del batch

            # Valutazione nell'Ambiente One-Shot
            env = Environment(raw_seqs, mode='sp')
            
            # evaluate applica la gap_matrix e calcola tutte le metriche
            reward, aligned_seqs_int, metrics = env.evaluate(gap_matrix)
            res = {
                'name': fasta_content.name,
                'QTY': metrics.get('QTY', len(raw_seqs)),
                'AL': metrics.get('AL', 0),
                'SP': metrics.get('SP', reward),
                'EM': metrics.get('EM', 0),
                'CS': metrics.get('CS', 0),
                'aligned': [decoder.decode(seq) for seq in aligned_seqs_int] 
            }
            inference_results.append(res)

    save_to_disk(inference_results, output_file, csv_file)
    print(f"\nInference completed successfully.")
    print(f"Report saved at: {output_file}")
    print(f"CSV saved at: {csv_file}\n\n")


def save_to_disk(results: List[dict], output_path: str, csv_path: str):
    with open(output_path, 'w') as f:
        for res in results:
            f.write(f"File: {res['name']}\n")
            f.write(f"Number of Sequences (QTY): {res['QTY']}\n")
            f.write(f"Alignment Length (AL): {res['AL']}\n")
            f.write(f"Sum of Pairs (SP): {res['SP']}\n")
            f.write(f"Exact Matches (EM): {res['EM']}\n")
            f.write(f"Column Score (CS): {res['CS']:.3f}\n")
            f.write("Alignment:\n")
            for s in res['aligned']:
                f.write(f"{s}\n")
            f.write("\n")

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File Name", "Number of Sequences (QTY)", "Alignment Length (AL)", "Sum of Pairs (SP)",
                         "Exact Matches (EM)", "Column Score (CS)"])
        for res in results:
            writer.writerow([res['name'], res['QTY'], res['AL'], res['SP'], res['EM'], res['CS']])

if __name__ == "__main__":
    csv_path = os.path.join(config.INFERENCE_CSV_PATH, 'GRPO/GRPO_results.csv')
    output_path = os.path.join(config.REPORTS_PATH, 'GRPO/GRPO_results.txt')
    run_inference(
        model_path = os.path.join(config.MODEL_WEIGHTS_PATH, 'grpo_model_epoch_100.pth'),
        data_folder = os.path.join(config.FASTA_FILES_PATH, config.DATASET_NAME),
        output_file = output_path,
        csv_file = csv_path
    )
