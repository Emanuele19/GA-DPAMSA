import torch
import os
from tqdm import tqdm
from typing import List

from DCNNMSA.dqn import DQNAgent
from DCNNMSA.env import Environment
from dataset_module import FastaDataset, SequenceEncoder, SequenceDecoder

import config, csv

def run_inference(model_path: str, data_folder: str, output_file: str, csv_file: str, n_seq: int = 3, vocab_size: int = 6):
    """
    Esegue l'inferenza caricando i file tramite FastaDataset e usando la classe FastaContent.
    """
    # 1. Setup Agente e Caricamento
    # n_sequences e vocab_size devono corrispondere a quelli usati in training
    agent = DQNAgent(n_sequences=n_seq, vocab_size=vocab_size, epsilon_start=0.0)
    
    # Carichiamo i pesi
    model_name = os.path.basename(model_path)
    model_dir = os.path.dirname(model_path)
    agent.load(model_name, model_dir)
    
    agent.policy_net.eval() # Fondamentale: disattiva dropout/batchnorm
    
    # 2. Setup Dataset
    # Nota: Passiamo l'encoder già presente nella policy_net o quello configurato
    dataset = FastaDataset(data_folder, SequenceEncoder(config.NUCLEOTIDE_ENCODING))
    
    inference_results = []

    print(f"Inizio Inferenza su {len(dataset)} file...")

    with torch.no_grad():
        for fasta_content in tqdm(dataset):
            # Creiamo l'ambiente con il tensore caricato pigramente da FastaContent
            env = Environment(fasta_content.tensor)
            state = env.reset()
            done = False
            
            # Loop di allineamento deterministico
            while not done:
                action = agent.select_action(state)
                state, reward, done = env.step(action)

            metrics = env.calculate_metrics()
            metrics['name'] = fasta_content.name
            metrics['aligned'] = env.get_alignment_as_strings(SequenceDecoder(config.NUCLEOTIDE_ENCODING))
            inference_results.append(metrics)

    # 4. Scrittura file di output
    save_to_disk(inference_results, output_file, csv_file) 

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

    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["File Name", "Number of Sequences (QTY)", "Alignment Length (AL)", "Sum of Pairs (SP)",
                         "Exact Matches (EM)", "Column Score (CS)"])
        for res in results:
            writer.writerow([res['name'], res['QTY'], res['AL'], res['SP'], res['EM'], res['CS']])

if __name__ == "__main__":
    run_inference(
        model_path = os.path.join(config.MODEL_WEIGHTS_PATH, 'msa_model_ep6000.pth'),
        data_folder = os.path.join(config.FASTA_FILES_PATH, 'synthetic_dataset_3x30bp'),
        output_file = os.path.join(config.REPORTS_PATH, 'DCNNMSA')
    )