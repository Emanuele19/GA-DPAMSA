import torch
import os
from tqdm import tqdm
from typing import List
import csv

from DCNN_BDDQN.dqn import DQNAgent
from DCNN_BDDQN.env import Environment
from dataset_module import FastaDataset, SequenceEncoder, SequenceDecoder
import config

def run_inference(model_path: str, data_folder: str, output_file: str, csv_file: str, n_seq: int = 3, vocab_size: int = 6):
    """
    Esegue l'inferenza caricando i file tramite FastaDataset e usando la classe FastaContent.
    Salva i risultati su disco (CSV e TXT).
    """
    # 1. Setup Agente e Caricamento
    agent = DQNAgent(n_sequences=n_seq, vocab_size=vocab_size, epsilon_start=0.0)
    
    # Carichiamo i pesi
    model_name = os.path.basename(model_path)
    model_dir = os.path.dirname(model_path)
    agent.load(model_name, model_dir)
    
    agent.policy_net.eval() # Fondamentale: disattiva dropout/batchnorm
    
    # 2. Setup Dataset
    dataset = FastaDataset(data_folder, SequenceEncoder(config.NUCLEOTIDE_ENCODING))
    inference_results = []

    print(f"Inizio Inferenza su {len(dataset)} file...")

    all_gaps_count = 0
    max_stasis_counter = 10
    gap_row_count = 0
    
    with torch.no_grad():
        for fasta_content in tqdm(dataset):
            # Creiamo l'ambiente con il tensore caricato pigramente da FastaContent
            env = Environment(fasta_content.tensor)
            state = env.reset()
            done = False
            deb = []
            deb_actions = []
            
            # Loop di allineamento deterministico
            while not done:
                action = agent.predict(state)

                state_input = state.unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_input)  # Output: (1, N, 2)
                probs = torch.softmax(q_values, dim=2) 
                deb.append(probs.squeeze(0))
                deb_actions.append(action.squeeze(0))
                
                state, reward, done, all_gaps_column = env.step(action)
                
                if all_gaps_column: 
                    all_gaps_count += 1
                    
                if any(all(s == torch.zeros_like(s)) for s in state):
                    gap_row_count += 1
                    break
                    
                if all_gaps_count > max_stasis_counter: 
                    done = True
                    all_gaps_count = 0
                    print(f"[DEBUG] stopped")
                    break

            metrics = env.calculate_metrics()
            metrics['name'] = fasta_content.name
            metrics['aligned'] = env.get_alignment_as_strings(SequenceDecoder(config.NUCLEOTIDE_DECODING))
            inference_results.append(metrics)

    # 4. Scrittura file di output (Fondamentale per il Benchmark!)
    save_to_disk(inference_results, output_file, csv_file) 
    print(f"\nInference completed successfully.")
    print(f"Report saved at: {output_file}")
    print(f"CSV saved at: {csv_file}\n\n")


def save_to_disk(results: List[dict], output_path: str, csv_path: str):
    # Assicurati che le cartelle esistano
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
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
            f.write(f"\n")

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # NOTA BENE: L'ordine delle colonne DEVE essere questo altrimenti il report HTML crasha!
        writer.writerow(["File Name", "Number of Sequences (QTY)", "Alignment Length (AL)", "Sum of Pairs (SP)",
                         "Exact Matches (EM)", "Column Score (CS)"])
        for res in results:
            writer.writerow([res['name'], res['QTY'], res['AL'], res['SP'], res['EM'], res['CS']])


if __name__ == "__main__":
    BASE_DS_NAME = 'hdf5_3x30_test_50'
    DATASET_NAME = f'orthodb_v12/{BASE_DS_NAME}'
    
    csv_path = os.path.join(config.INFERENCE_CSV_PATH, 'DCNN_BDDQNMSA/DCNN_BDDQNMSA_results.csv')
    out_path = os.path.join(config.REPORTS_PATH, 'DCNN_BDDQNMSA/DCNN_BDDQNMSA_results.txt')
    
    run_inference(
        model_path = os.path.join(config.PROJECT_ROOT, 'DCNN_BDDQNMSA/weights/msa_model_ep18999.pth'),
        data_folder = os.path.join(config.FASTA_FILES_PATH, DATASET_NAME),
        output_file = out_path,
        csv_file = csv_path
    )