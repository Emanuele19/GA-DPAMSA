from tqdm import tqdm
import config, os

from env import Environment
from dqn import DQNAgent
from dataset_module import FastaDataset, SequenceEncoder

from model_utils import setup_logger
logger = setup_logger()


def train():
    # --- CONFIGURAZIONE ---
    SAVE_FREQ = 50          
    MIN_REPLAY_SIZE = 1000  
    
    # Inizializzazione Dataset
    dataset = FastaDataset(
        folder_path=os.path.join(config.FASTA_FILES_PATH, 'dataset1_3x30bp'), 
        num_workers=4,
        encoder=SequenceEncoder({'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4, 'P': 5}))
    
    # --- CHECK DIMENSIONE DATASET ---
    # Calcoliamo quanti file abbiamo effettivamente a disposizione
    available_files = len(dataset)
    
    if available_files == 0:
        logger.error("Errore: Il dataset è vuoto.")
        return

    if config.MAX_EPISODE > available_files:
        logger.warning(f"Richiesti {config.MAX_EPISODE} episodi, ma il dataset contiene solo {available_files} file.")
        logger.warning(f"Il training verrà eseguito per {available_files} episodi.\n")
        num_episodes = available_files
    else:
        num_episodes = config.MAX_EPISODE

    # Inizializzazione Agente
    agent = DQNAgent(n_sequences=3, vocab_size=6)

    # Riempi il replay buffer di decisioni casuali
    fill_buffer(agent, dataset, MIN_REPLAY_SIZE)

    # Trainining loop
    logger.info(f"\nInizio Addestramento su {num_episodes} file...")
    
    # Iteriamo direttamente sul dataset per il numero di episodi calcolato
    # Usiamo enumerate per tenere traccia del conteggio
    for episode, fasta_content in enumerate(dataset, 1):
        if episode > num_episodes:
            break
            
        env = Environment(fasta_content.tensor)
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.learn(config.BATCH_SIZE)
            
            state = next_state
            episode_reward += reward
        
        # Aggiornamenti periodici
        agent.update_epsilon()
        
        if episode % config.UPDATE_ITERATION == 0:
            agent.update_target_network()

        if episode % 10 == 0:
            logger.info(f"Ep {episode:4d}/{num_episodes} | Rew: {episode_reward:7.2f} | Eps: {agent.epsilon:.3f}")

        if episode % SAVE_FREQ == 0:
            agent.save(f"msa_model_ep{episode}.pth")

    logger.info(f"\nTraining completato. Eseguiti {episode} episodi.")

def fill_buffer(agent: DQNAgent, dataset: FastaDataset, min_size: int):
    logger.info(f"Inizio Warm-up (Target: {min_size} esperienze)...")
    pbar = tqdm(total=min_size, desc="Filling Buffer")
    
    # Per il warm-up usiamo un iteratore temporaneo per non "consumare" quello del training
    warmup_iter = iter(dataset)
    while len(agent.memory) < min_size:
        try:
            fasta_content = next(warmup_iter)
        except StopIteration:
            # Se i file finiscono durante il warm-up, ricominciamo l'iteratore
            warmup_iter = iter(dataset)
            fasta_content = next(warmup_iter)
            
        env = Environment(fasta_content.tensor)
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