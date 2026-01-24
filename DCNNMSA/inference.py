import torch
import os
from tqdm import tqdm
from typing import List

from dqn import DQNAgent
from dataset_module import FastaDataset, SequenceEncoder, SequenceDecoder
from env import Environment

import config

def run_inference(model_path: str, data_folder: str, output_file: str, n_seq: int = 3, vocab_size: int = 6):
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
            
            # 3. Estrazione dati finali
            # Assumiamo che env.get_final_alignment() restituisca List[str]
            # e env.calculate_cs() restituisca il Column Score
            aligned_seqs = env.get_alignment_as_strings()
            cs_score = env.calculate_total_cs()
            perc_cs_score = env.get_cs_percentage() 

            decoder = SequenceDecoder(config.NUCLEOTIDE_ENCODING)
            original_sequence = decoder.decode_batch(fasta_content.tensor)

            inference_results.append({
                'name': fasta_content.name,
                'original': original_sequence,
                'aligned': aligned_seqs,
                'cs': cs_score,
                'perc_cs': perc_cs_score
            })

    # 4. Scrittura file di output
    save_to_disk(inference_results, output_file) 

def save_to_disk(results: List[dict], output_path: str):
    with open(output_path, 'w') as f:
        for res in results:
            f.write(f"FILE: {res['name']}\n")
            f.write(f"Column Score (CS): {res['cs']:.4f} | ({res['perc_cs']:.2f}%)\n")
            f.write("-" * 20 + "\n")
            f.write("ALIGNED SEQUENCES:\n")
            for s in res['aligned']:
                f.write(f"{s}\n")
            f.write("\n" + "="*40 + "\n\n")

if __name__ == "__main__":
    run_inference(
        model_path = os.path.join(config.MODEL_WEIGHTS_PATH, 'msa_model_ep6000.pth'),
        data_folder = os.path.join(config.FASTA_FILES_PATH, 'synthetic_dataset_3x30bp'),
        output_file = os.path.join(config.REPORTS_PATH, 'DCNNMSA')
    )