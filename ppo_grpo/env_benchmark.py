import time
import torch
import random
import numpy as np

# --- IMPORTS ---
# Ensure these paths match your project structure
from ppo_grpo.env import Environment as CPUEnvironment
from ppo_grpo.vect_env import VectorizedEnvironment


# --- MOCK CONFIGURATION ---
class MockConfig:
    GAP_TOKEN = 5
    PADDING_TOKEN = 0
    MATCH_REWARD = 4.0
    MISMATCH_PENALTY = -4.0
    GAP_PENALTY = -2.0


# --- DATA GENERATION UTILITY ---
def generate_synthetic_data(batch_size, num_rows, seq_len, device):
    """
    Generates random data for both Tensor (GPU) and List (CPU) formats.
    """
    # 1. GPU Data
    # Random DNA (values 1-4)
    raw_seqs_gpu = torch.randint(1, 5, (batch_size, num_rows, seq_len), device=device)
    # Random Gaps (values 0-3)
    gap_counts_gpu = torch.randint(0, 3, (batch_size, num_rows, seq_len), device=device)

    # 2. CPU Data (Conversion for fair comparison)
    # We convert tensors to lists outside the timing loop to measure only execution speed.
    cpu_tensor_seq = raw_seqs_gpu.cpu()
    cpu_tensor_gap = gap_counts_gpu.cpu()

    cpu_tasks = []
    for b in range(batch_size):
        raw_rows = cpu_tensor_seq[b].tolist()
        gap_rows = cpu_tensor_gap[b].tolist()
        cpu_tasks.append((raw_rows, gap_rows))

    return raw_seqs_gpu, gap_counts_gpu, cpu_tasks


# --- BENCHMARK ENGINE ---
def run_benchmark():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Benchmark on: {device}")

    config = MockConfig()

    # Instantiate GPU Environment (Reuse instance)
    env_gpu = VectorizedEnvironment(config, device)

    # Define scales to test
    # Increasing Batch Sizes to demonstrate GPU scaling
    batch_sizes = [4, 16, 64, 256, 1024]

    # Fixed parameters
    NUM_ROWS = 3  # Typical MSA block size
    SEQ_LEN = 30  # Sequence length
    ITERATIONS = 10  # Number of runs to average

    print(f"\nSettings: Rows={NUM_ROWS}, SeqLen={SEQ_LEN}, Averaged over {ITERATIONS} runs.\n")

    # WARM-UP GPU
    # The first CUDA call is always slow (context allocation). We do a dummy run.
    dummy_seq = torch.randint(1, 5, (1, 3, 10), device=device)
    dummy_gap = torch.zeros_like(dummy_seq)
    env_gpu.evaluate_batch(dummy_seq, dummy_gap)
    torch.cuda.synchronize()

    # Print Table Header
    print(f"{'Batch Size':<12} | {'CPU Time (Seq)':<16} | {'GPU Time (Par)':<16} | {'Speedup':<10}")
    print("-" * 65)

    for bs in batch_sizes:
        # 1. Generate Data
        gpu_seq, gpu_gap, cpu_tasks = generate_synthetic_data(bs, NUM_ROWS, SEQ_LEN, device)

        # -----------------------------
        # 2. TEST CPU (Sequential)
        # -----------------------------
        # We simulate the loop of the current Trainer: iterating over the batch
        start_cpu = time.time()
        for _ in range(ITERATIONS):
            # Iterate through the batch (Trainer simulation)
            for raw_data, gap_data in cpu_tasks:
                # Instantiation + Evaluation
                env_cpu = CPUEnvironment(raw_data, mode='sp')
                env_cpu.evaluate(gap_data)

        end_cpu = time.time()
        avg_cpu_time = (end_cpu - start_cpu) / ITERATIONS

        # -----------------------------
        # 3. TEST GPU (Parallel)
        # -----------------------------
        torch.cuda.synchronize()  # Wait for everything to settle
        start_gpu = time.time()

        for _ in range(ITERATIONS):
            # Single vectorized call
            # Note: We pass the entire batch at once
            env_gpu.evaluate_batch(gpu_seq, gpu_gap)

        torch.cuda.synchronize()  # Wait for GPU to actually finish!
        end_gpu = time.time()
        avg_gpu_time = (end_gpu - start_gpu) / ITERATIONS

        # -----------------------------
        # 4. Statistics & Output
        # -----------------------------
        # Calculate Speedup
        # Avoid division by zero
        speedup = avg_cpu_time / (avg_gpu_time + 1e-9)

        # Print Row
        print(f"{bs:<12} | {avg_cpu_time * 1000:6.2f} ms       | {avg_gpu_time * 1000:6.2f} ms       | {speedup:6.1f}x")

    print("-" * 65)
    print("\nBenchmark Complete.")
    print("- CPU Time: Includes Python loop 'for i in batch'.")
    print("- GPU Time: Pure tensor computation time.")
    print("- Speedup: How many times faster the GPU implementation is.")


if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        print(f"Benchmark Error: {e}")
        # Helpful hint if imports fail
        import sys

        print(f"\nMake sure you are running this from the project root.")
        print(f"Current path: {sys.path[0]}")