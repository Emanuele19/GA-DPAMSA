import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


import config
from utils import setup_logger
from dataset_module import MSADataset

from agent.backbone.dcnn import DCNNBackbone
from agent.output_adapter.gaussian import GaussianGapAdapter
from agent.actor.linear import MSA_Actor
from agent.critic.linear import MSA_Critic
from agent.ppo_agent import PPO_Agent
from agent.grpo_agent import GRPO_Agent

from trainer.ppo import PPOTrainer
from trainer.grpo import GRPOTrainer

from data.integer import IntegerStatePreprocessor


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
    backbone = DCNNBackbone(
        num_rows=config.AGENT_WINDOW_ROW,  # e.g., 3 sequences per block
        vocab_size=len(config.NUCLEOTIDES_MAP) + 1,  # +1 for safe padding handling
        embedding_dim=64,
        hidden_dim=12
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
        critic_backbone = DCNNBackbone(
            num_rows=config.AGENT_WINDOW_ROW,
            vocab_size=len(config.NUCLEOTIDES_MAP) + 1,
            embedding_dim=64,
            hidden_dim=128
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
            padding_idx=pad_idx
        )

    else:
        raise ValueError(f"Unknown Algorithm in config: {config.ALGO}")

    # 5. TRAINING LOOP
    logger.info(f"Starting Training Loop for {config.MAX_EPOCH} epochs...")

    try:
        for epoch in range(config.MAX_EPOCH):
            logger.info(f"--- Epoch {epoch + 1}/{config.MAX_EPOCH} ---")

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