import csv
import os
import subprocess
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import webbrowser

import config
from DPAMSA.dqn import DQN
from DPAMSA.env import Environment
import utils

import datasets.training_dataset.synthetic_dataset_4x101bp as inference_dataset
import datasets.training_dataset.zhang_dataset_3x30 as training_dataset

from dataset_module import FastaContent, FastaDataset

"""
DPAMSA Main Script

This script serves as the main entry point for running the DPAMSA framework. 
It provides functionalities for training a reinforcement learning model, 
running inference using a pre-trained model, and managing datasets.

Key Features:
- Loads datasets and configurations.
- Trains a Deep Q-Network (DQN) for sequence alignment.
- Runs inference to generate alignments using a trained model.
- Saves results as reports and CSV files.
- Provides an interactive menu for user selection.

Author (legacy): https://github.com/ZhangLab312/DPAMSA
Co-Author (improved): https://github.com/FLaTNNBio/GA-DPAMSA
"""


TRAINING_DATASET = training_dataset
INFERENCE_DATASET = inference_dataset
INFERENCE_MODEL = 'model_3x30'


def output_parameters():
    """
    Print the key DPAMSA configuration parameters.
    """
    print("---- DPAMSA parameters ---")
    print("Gap penalty: {}".format(config.GAP_PENALTY))
    print("Mismatch penalty: {}".format(config.MISMATCH_PENALTY))
    print("Match reward: {}".format(config.MATCH_REWARD))
    print("Episode: {}".format(config.MAX_EPISODE))
    print("Batch size: {}".format(config.BATCH_SIZE))
    print("Replay memory size: {}".format(config.REPLAY_MEMORY_SIZE))
    print("Alpha: {}".format(config.ALPHA))
    print("Epsilon: {}".format(config.EPSILON))
    print("Gamma: {}".format(config.GAMMA))
    print("Delta: {}".format(config.DELTA))
    print("Decrement iteration: {}".format(config.DECREMENT_ITERATION))
    print("Update iteration: {}".format(config.UPDATE_ITERATION))
    print("Device: {}".format(config.DEVICE_NAME))
    print('\n')


def open_tensorboard(log_dir):
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

        # Start TensorBoard as a background process
        tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6006"])

        # Wait a few seconds to ensure TensorBoard starts properly
        time.sleep(3)

        # Open TensorBoard in the default web browser
        webbrowser.open("http://localhost:6006")

        return tensorboard_process

    except Exception as e:
        print(f"⚠️ Error starting TensorBoard: {e}")
        return None


def train(dataset=TRAINING_DATASET, start=0, end=-1, model_path='new_model_3x30'):
    """
    Train a reinforcement learning model on the given dataset.

    Parameters:
    -----------
    - dataset (module): The dataset module containing sequences.
    - start (int): Index to start processing datasets.
    - end (int): Index to stop processing datasets (-1 for all).
    - model_path (str): Path to save the trained model.
    """
    output_parameters()

    # Create SummaryWriter instances for each dataset
    writers = {}
    log_dir = os.path.join(config.RUNS_PATH, os.path.splitext(dataset.file_name)[0])
    for name in dataset.datasets:
        writers[name] = SummaryWriter(os.path.join(log_dir, name))

        # Write an initial log entry so TensorBoard detects the logs immediately
        writers[name].add_scalar('Training/Loss', 0, 0)
        writers[name].add_scalar('Training/Reward', 0, 0)
        writers[name].add_scalar('Training/Steps', 0, 0)
        writers[name].add_scalar('Training/Epsilon', config.EPSILON, 0)
        writers[name].add_scalar('Metrics/SP', 0, 0)
        writers[name].add_scalar('Metrics/CS', 0, 0)
        writers[name].flush()  # Force immediate write to disk

    # Automatically launch TensorBoard
    _ = open_tensorboard(log_dir)

    # Get the subset of datasets to process
    datasets_to_process = dataset.datasets[start:end if end != -1 else len(dataset.datasets)]
    for index, name in enumerate(datasets_to_process, start):

        if not hasattr(dataset, name):
            continue
        seqs = getattr(dataset, name)

        # Initialize environment and DQN agent
        env = Environment(seqs)
        agent = DQN(env.action_number, env.row, env.max_len, env.max_len * env.max_reward)

        # Load pre-trained model if available
        try:
            agent.load(model_path + '.pth')
        except:
            pass

        # Early Stopping Parameters
        best_avg_reward = -float('inf')  # Best moving average reward observed
        no_improve_count = 0             # Count of episodes without improvement
        early_stopping_patience = 200    # Threshold for stopping training
        reward_history = []              # Store recent rewards for analysis

        # Create a single tqdm progress bar
        pbar = tqdm(total=config.MAX_EPISODE, desc=f'Training on {name}', position=0, leave=True, dynamic_ncols=True)

        # Training loop
        for episode in range(config.MAX_EPISODE):
            # Reset the alignment state to its initial condition
            state = env.reset()

            # Initialize trackers
            episode_reward = 0
            episode_loss = 0
            steps = 0

            while True:
                # Select an action using the agent's policy
                action = agent.select(state)

                # Execute action in the environment
                reward, next_state, done = env.step(action)

                # Store transition in replay memory
                agent.replay_memory.push((state, next_state, action, reward, done))

                # Update the model using replay memory
                loss = agent.update()

                # Accumulate episode statistics
                episode_reward += reward
                if loss is not None:  # Check if loss is not None
                    episode_loss += loss
                steps += 1

                if done == 0:  # End of episode
                    break
                state = next_state

            # Update epsilon (exploration-exploitation balance)
            agent.update_epsilon()

            # Training metrics calculation
            sp_score = env.calc_score()
            alignment_length = len(env.aligned[0])
            exact_matches = env.calc_exact_matched()
            column_score = exact_matches / alignment_length

            # Update tqdm progress bar with relevant info
            pbar.set_postfix({
                "LAST EP. STATS": f"[ Loss: {episode_loss:.2f}, Reward: {episode_reward:.2f}, Steps: {steps}, Epsilon: {agent.current_epsilon:.2f} ]"
            })
            pbar.update(1)

            # TensorBoard Logging
            writers[name].add_scalar('Training/Loss', episode_loss, episode)
            writers[name].add_scalar('Training/Reward', episode_reward, episode)
            writers[name].add_scalar('Training/Steps', steps, episode)
            writers[name].add_scalar('Training/Epsilon', agent.current_epsilon, episode)
            writers[name].add_scalar('Metrics/SP', sp_score, episode)
            writers[name].add_scalar('Metrics/CS', column_score, episode)

            # Early Stopping: Check reward improvement over last 100 episodes
            reward_history.append(episode_reward)
            if len(reward_history) > 100:
                reward_history.pop(0)  # Keep only last 100 episodes
                avg_recent_reward = sum(reward_history) / len(reward_history)

                if avg_recent_reward > best_avg_reward:
                    best_avg_reward = avg_recent_reward
                    no_improve_count = 0   # Reset counter if improvement is observed
                else:
                    no_improve_count += 1  # # Increment if no improvement

                # Stop training if no improvement for "early_stopping_patience" episodes
                if no_improve_count >= early_stopping_patience:
                    print(f"Early stopping activated. No improvement for {early_stopping_patience} episodes.")
                    break

        # Close progress bar after training on dataset
        pbar.close()

        # Close TensorBoard writer for the dataset
        writers[name].close()

        # Save the trained model
        agent.save(model_path)

    print(f"\nTraining completed successfully.")




def inference(dataset: FastaDataset, start=0, end=-1, model_path=INFERENCE_MODEL, truncate_file=True):
    """
    Run inference using a pre-trained model on a given dataset.

    Parameters:
    - dataset (module): The dataset module containing sequences.
    - start (int): Index to start processing datasets.
    - end (int): Index to stop processing datasets (-1 for all).
    - model_path (str): Path to the pre-trained model.
    - truncate_file (bool): Whether to overwrite report files.
    """
    output_parameters()

    tag = dataset.name
    report_file_name = os.path.join(config.DPAMSA_REPORTS_PATH, f"{tag}.txt")
    csv_file_name = os.path.join(config.DPAMSA_INF_CSV_PATH, f"{tag}_DPAMSA_results.csv")

    # Create or truncate results files
    if truncate_file:
        with open(report_file_name, 'w'):
            pass
        with open(csv_file_name, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["File Name", "Number of Sequences (QTY)", "Alignment Length (AL)", "Sum of Pairs (SP)",
                             "Exact Matches (EM)", "Column Score (CS)"])

    datasets_to_process:list[FastaContent] = dataset[start:end if end != -1 else len(dataset)]

    for index, fasta_content in enumerate(tqdm(datasets_to_process, desc="Processing Datasets"), start):
        seqs = fasta_content.sequences

        env = Environment(seqs)
        agent = DQN(env.action_number, env.row, env.max_len, env.max_len * env.max_reward)
        agent.load(model_path)
        state = env.reset()

        while True:
            action = agent.predict(state)
            _, next_state, done = env.step(action)
            state = next_state
            if 0 == done:
                break

        env.padding()

        # Compute metrics
        metrics = utils.calculate_metrics(env)

        # Create report
        report = (
            f"File: {fasta_content.name}\n"
            f"Number of Sequences (QTY): {metrics['QTY']}\n"
            f"Alignment Length (AL): {metrics['AL']}\n"
            f"Sum of Pairs (SP): {metrics['SP']}\n"
            f"Exact Matches (EM): {metrics['EM']}\n"
            f"Column Score (CS): {metrics['CS']:.3f}\n"
            f"Alignment:\n{env.get_alignment()}\n\n"
        )

        # Save results to files
        with open(report_file_name, 'a') as report_file:
            report_file.write(report)

        with open(csv_file_name, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([fasta_content.name, metrics['QTY'], metrics['AL'], metrics['SP'], metrics['EM'], metrics['CS']])

    print(f"\nInference completed successfully.")
    print(f"Report saved at: {report_file_name}")
    print(f"CSV saved at: {csv_file_name}\n\n")


def menu():
    """
    Interactive menu for training/inference mode selection.
    """
    while True:
        print("\n====== DPAMSA MENU ======")
        output_parameters()
        print(f"Dataset loaded for Training: {TRAINING_DATASET.file_name}")
        print(f"Dataset loaded for Inference: {INFERENCE_DATASET.file_name}\n\n")
        print("1 - Train the model")
        print("2 - Run inference")
        print("3 - Exit\n")

        choice = input("Select an option (1/2/3): ").strip()

        if choice == "1":
            print("\nStarting training...")
            config.DEVICE = torch.device(config.DEVICE_NAME)
            train()

        elif choice == "2":
            print(f"\nModel selected for inference: {INFERENCE_MODEL}")
            confirm = input("Do you want to proceed with inference? (yes/no): ").strip().lower()
            if confirm == "yes":
                print("\nStarting inference...")
                inference()
            else:
                print("\nInference canceled.")

        elif choice == "3":
            print("\nExiting program. Goodbye!")
            break

        else:
            print("\nInvalid choice, please enter 1, 2, or 3.")


if __name__ == "__main__":
    menu()
