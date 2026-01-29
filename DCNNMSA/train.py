from tqdm import tqdm
import config, os

from env import Environment
from dqn import DQNAgent
from dataset_module import MSADataset, SequenceEncoder
from torch.utils.tensorboard import SummaryWriter

from model_utils import setup_logger
logger = setup_logger()

gap_idx = config.NUCLEOTIDE_ENCODING['-']
pad_idx = config.NUCLEOTIDE_ENCODING['P']

def open_tensorboard(log_dir):
    import subprocess, time, webbrowser
    """
    Launch TensorBoard and open it in the default web browser.

    Parameters:
    -----------
    - log_dir (str): Path to the directory where TensorBoard logs are stored.

    Returns:
    --------
    - subprocess.Popen: The process running TensorBoard (can be terminated later).
    """
    try:
        print("🚀 Starting TensorBoard on http://localhost:6006...")
        tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6006"])
        time.sleep(3)
        webbrowser.open("http://localhost:6006")

        return tensorboard_process

    except Exception as e:
        print(f"⚠️ Error starting TensorBoard: {e}")
        return None

def train():
    # --- CONFIGURAZIONE ---
    SAVE_FREQ = 1000          
    MIN_REPLAY_SIZE = 1000  
    TENSORBOARD_PATH = "runs/msa_dqn_experiment"
    
    # Inizializzazione Dataset
    dataset = MSADataset(os.path.join(config.FASTA_FILES_PATH, 'orthodb_v12/hdf5_3x30.h5'))
    
    # --- CHECK DIMENSIONE DATASET ---
    # Calcoliamo quanti file abbiamo effettivamente a disposizione
    number_of_alignments, N, W = dataset.shape
    
    if number_of_alignments == 0:
        logger.error("Errore: Il dataset è vuoto.")
        return

    if config.MAX_EPISODE > number_of_alignments:
        logger.warning(f"Richiesti {config.MAX_EPISODE} episodi, ma il dataset contiene solo {number_of_alignments} file.")
        logger.warning(f"Il training verrà eseguito per {number_of_alignments} episodi.\n")
        num_episodes = number_of_alignments
    else:
        num_episodes = config.MAX_EPISODE

    writer = SummaryWriter(log_dir=TENSORBOARD_PATH)
    open_tensorboard(TENSORBOARD_PATH)

    # Inizializzazione Agente
    agent = DQNAgent(n_sequences=N, vocab_size=6)

    # Riempi il replay buffer di decisioni casuali
    fill_buffer(agent, dataset, W, MIN_REPLAY_SIZE)

    # Trainining loop
    logger.info(f"\nInizio Addestramento su {num_episodes} file...")
    
    # Iteriamo direttamente sul dataset per il numero di episodi calcolato
    # Usiamo enumerate per tenere traccia del conteggio
    p_bar = tqdm(total=num_episodes, desc="Training")
    for episode, alignment in enumerate(dataset):
        if episode > num_episodes:
            break
            
        env = Environment(alignment.sequences, fixed_size=W, 
                          gap_idx=gap_idx, pad_idx=pad_idx)
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            learn_result = agent.learn(config.BATCH_SIZE)
            if learn_result:
                loss, gradient_norm = learn_result
                episode_loss.append(loss)
                writer.add_scalar("Debug/Gradient_Norm_Step", gradient_norm, episode)
            
            state = next_state
            episode_reward += reward

        writer.add_scalar("Training/Reward", episode_reward, episode)
        writer.add_scalar("Training/Epsilon", agent.epsilon, episode)
        writer.add_scalar("Metrics/CS", env.calculate_total_cs(), episode)
        writer.add_scalar("Metrics/SP", env.calc_sp_score(), episode)
        if episode_loss:
            writer.add_scalar("Metrics/Avg_loss", sum(episode_loss) / len(episode_loss), episode)

        if episode % 100 == 0:
            for name, param in agent.policy_net.named_parameters():
                writer.add_histogram(f"Weights/{name}", param, episode)
                if param.grad is not None:
                    writer.add_histogram(f"Gradients/{name}", param.grad, episode)
        
        # Aggiornamenti periodici
        agent.update_epsilon()
        
        if episode % config.UPDATE_ITERATION == 0:
            agent.update_target_network()

        if episode % SAVE_FREQ == 0 and episode != 0:
            agent.save(f"msa_model_ep{episode}.pth")

        p_bar.update(1)

    writer.close()
    logger.info(f"\nTraining completato. Eseguiti {episode} episodi.")

def fill_buffer(agent: DQNAgent, dataset: MSADataset, fixed_size: int, min_size: int):
    logger.info(f"Inizio Warm-up (Target: {min_size} esperienze)...")
    pbar = tqdm(total=min_size, desc="Filling Buffer")
    
    # Per il warm-up usiamo un iteratore temporaneo per non "consumare" quello del training
    warmup_iter = iter(dataset)
    while len(agent.memory) < min_size:
        try:
            alignment = next(warmup_iter)
        except StopIteration:
            # Se i file finiscono durante il warm-up, ricominciamo l'iteratore
            warmup_iter = iter(dataset)
            alignment = next(warmup_iter)
            
        env = Environment(alignment.sequences, fixed_size=fixed_size,
                          gap_idx=gap_idx, pad_idx=pad_idx)
        state = env.reset()
        done = False
        while not done and len(agent.memory) < min_size:
            action = agent.select_action(state) 
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            pbar.update(1)
    pbar.close()

if __name__ == "__main__":
    train()