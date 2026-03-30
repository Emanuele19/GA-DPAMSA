import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


import config
from utils import setup_logger
from dataset_module import MSADataset

from ppo_grpo.agent.backbone.new import RobustBackbone
from ppo_grpo.agent.output_adapter.gaussian import GaussianGapAdapter
from ppo_grpo.agent.actor.mlp import MSA_Actor
from ppo_grpo.agent.critic.linear import MSA_Critic
from ppo_grpo.agent.ppo_agent import PPO_Agent
from ppo_grpo.agent.grpo_agent import GRPO_Agent

from ppo_grpo.trainer.ppo import PPOTrainer
from ppo_grpo.trainer.grpo import GRPOTrainer

from ppo_grpo.data.integer import IntegerStatePreprocessor


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
        print("Starting TensorBoard on http://localhost:6006...")
        tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6006"])
        time.sleep(3)
        webbrowser.open("http://localhost:6006")

        return tensorboard_process

    except Exception as e:
        print(f"Error starting TensorBoard: {e}")
        return None
    

def get_collate_fn():
    """
    Custom collate function for the DataLoader.

    Standard PyTorch default_collate tries to stack inputs into tensors immediately.
    Since our MSADataset yields complex objects (MSAAlignment) or jagged arrays
    that might need specific padding logic handled by the Trainer, we simply
    return the batch as a list.

    The BaseTrainer._prepare_batch() method will handle the conversion to
    padded tensors later.

    Returns:
        function: Identity function that returns the raw list of batch items.
    """

    def identity_collate(batch):
        return batch

    return identity_collate


def main():
    # 1. INITIAL SETUP
    # Initialize the logger (from utils.py)
    logger = setup_logger()

    # Detect hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting Training on device: {device}")
    logger.info(f"selected Algorithm: {config.ALGO} (One-Shot Generation)")

    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=config.TENSORBOARD_PATH)

    #Start TensorBoard 
    tb_process = open_tensorboard(config.TENSORBOARD_PATH)
    # 2. DATASET LOADING
    # We use the HDF5 dataset which is efficient for large bio-data
    dataset_path = os.path.join(config.FASTA_FILES_PATH, 'orthodb_v12/hdf5_3x30.h5')

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at: {dataset_path}")
        return

    # Load the HDF5 Dataset
    dataset = MSADataset(dataset_path)
    logger.info(f"Dataset loaded: {len(dataset)} alignment blocks.")

    # Initialize DataLoader
    # num_workers=0 is safer with HDF5 to avoid file locking issues in multiprocessing
    train_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=get_collate_fn()
    )

    # 3. MODEL CONSTRUCTION (Modular Assembly)
    logger.info("Building Model Architecture...")

    # A. Preprocessor
    # Uses Integer preprocessor with sanitization
    preprocessor = IntegerStatePreprocessor(
        config,
        device=device,
    )

    # A. Backbone (The 'Eye')
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

    # C. Actor (The 'Brain')
    # Connects Backbone to Adapter
    actor = MSA_Actor(backbone, adapter)

    # 4. AGENT & TRAINER SELECTION
    # We choose the strategy based on config.ALGO ('GRPO' or 'PPO')

    # Retrieve padding index safely from config
    pad_idx = config.NUCLEOTIDES_MAP.get('P', 0)

    if config.ALGO == 'GRPO':
        logger.info("Initializing GRPO Agent (Generative Rejection Policy)...")

        # GRPO only needs the Actor (Policy)
        agent = GRPO_Agent(actor, lr=config.LR, device=device)

        # Initialize the GRPO Trainer
        trainer = GRPOTrainer(
            agent=agent,
            preprocessor=preprocessor,
            group_size=config.GRPO_GROUP_SIZE,  # e.g., Generate 8 variants per input
            env_mode='sp',  # Use Sum-of-Pairs scoring
            logger=logger,
            writer=writer,
            output_dir=config.MODEL_WEIGHTS_PATH,
            padding_idx=pad_idx,
            config=config,
        )

    elif config.ALGO == 'PPO':
        logger.info("Initializing PPO Agent (Actor-Critic)...")

        # PPO requires a Critic (Value Function)
        # We create a separate Backbone for the Critic to ensure stability
        critic_backbone = RobustBackbone(
            num_rows=config.AGENT_WINDOW_ROW,
            vocab_size=len(config.NUCLEOTIDES_MAP) + 1,
            embedding_dim=128,
            hidden_dim=128,
            num_layers=4,
        )
        critic = MSA_Critic(critic_backbone)

        # Initialize PPO Agent with both networks
        agent = PPO_Agent(actor, critic, lr=config.LR, device=device)

        # Initialize PPO Trainer
        trainer = PPOTrainer(
            agent=agent,
            preprocessor=preprocessor,
            clip_eps=0.2,  # PPO Clipping Epsilon
            ppo_epochs=4,  # Number of update epochs per batch
            env_mode='sp',
            logger=logger,
            writer=writer,
            output_dir=config.MODEL_WEIGHTS_PATH,
            padding_idx=pad_idx,
            config=config
        )

    else:
        raise ValueError(f"Unknown Algorithm in config: {config.ALGO}")

    # ---------------------------------------------------------
    # 4.5 RESUME FROM CHECKPOINT (Resume Training)
    # ---------------------------------------------------------
    # Set this to True to enable resuming from a checkpoint. Make sure to specify the correct epoch number and checkpoint file.
    RESUME_TRAINING = True
    START_EPOCH = 30  # change this to the epoch number you want to resume from 
    CHECKPOINT_FILE = f"checkpoint_ep{START_EPOCH}.pt" 
    
    if RESUME_TRAINING:
        checkpoint_path = os.path.join(config.MODEL_WEIGHTS_PATH, CHECKPOINT_FILE)
        if os.path.exists(checkpoint_path):
            logger.info(f"🔄 Ripristino addestramento dal checkpoint: {checkpoint_path}")
            agent.load(checkpoint_path)
        else:
            logger.warning(f"⚠️ Checkpoint {checkpoint_path} non trovato! Inizio da zero.")
            START_EPOCH = 0
    else:
        START_EPOCH = 0

    # 5. TRAINING LOOP
    logger.info(f"Starting Training Loop for {config.MAX_EPOCH - START_EPOCH} epochs...")

    try:
        for epoch in (range(config.MAX_EPOCH - START_EPOCH)):
            logger.info(f"--- Epoch {epoch + 1}/{config.MAX_EPOCH - START_EPOCH} ---")

            # The trainer handles the iteration over the dataloader
            # and performs the specific optimization steps (GRPO or PPO)
            trainer.train_epoch(train_loader, epoch_idx=epoch)

            # Periodic Checkpointing
            if (epoch + 1) % config.SAVE_FREQ == 0:
                trainer.save_checkpoint(f"checkpoint_ep{epoch + 1}.pt")

    except KeyboardInterrupt:
        logger.warning("Training interrupted manually (Ctrl+C). Saving emergency checkpoint...")
        trainer.save_checkpoint("interrupted_save.pt")

    # 6. CLEANUP & FINAL SAVE
    logger.info("Saving final model...")
    trainer.save_checkpoint("final_model.pt")

    writer.close()

    # Close HDF5 file handle if supported
    if hasattr(dataset, 'close'):
        dataset.close()

    logger.info("Training Complete.")


if __name__ == "__main__":
    main()