import torch
import os
from tqdm import tqdm
from typing import List

from DCNNMSA.dqn import DQNAgent
from DCNNMSA.env import Environment
from dataset_module import FastaDataset, SequenceEncoder, SequenceDecoder

import config, csv

def run_inference(model_path: str, data_folder: str, n_seq: int = 3, vocab_size: int = 6):
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
            metrics['aligned'] = env.get_alignment_as_strings(SequenceDecoder(config.NUCLEOTIDE_DECODING))
            inference_results.append(metrics)

    return inference_results


if __name__ == "__main__":
    from utils import save_to_disk
    csv_path = os.path.join(config.INFERENCE_CSV_PATH, 'DCNNMSA/DCNNMSA_results.csv')
    out_path = os.path.join(config.REPORTS_PATH, 'DCNNMSA/DCNNMSA_results.txt')
    results = run_inference(
        model_path = os.path.join(config.MODEL_WEIGHTS_PATH, 'msa_model_ep18999.pth'),
        data_folder = os.path.join(config.FASTA_FILES_PATH, 'orthodb_v12/hdf5_3x30_test_10')
    )
    save_to_disk(results, out_path, csv_path)