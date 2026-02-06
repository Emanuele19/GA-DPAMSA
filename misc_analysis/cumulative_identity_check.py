import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIGURAZIONE ---
HDF5_PATH = "../datasets/fasta_files/orthodb_v12/hdf5_3x30.h5"
PLOT_OUTPUT = "cumulative_identity.png"
from dataset_module import MSADataset


def calculate_block_identity(matrix: np.ndarray) -> float:
    """Calcola l'identità media a coppie."""
    n_rows, width = matrix.shape
    if n_rows < 2: return 1.0

    scores = []
    pairs = list(itertools.combinations(range(n_rows), 2))
    for r1, r2 in pairs:
        matches = np.sum(matrix[r1] == matrix[r2])
        scores.append(matches / width)
    return np.mean(scores)


def print_cumulative_table(all_scores):
    """Stampa la tabella cumulativa inversa (> Soglia)."""
    total = len(all_scores)

    print(f"\n{'THRESHOLD':<15} | {'COUNT (> X)':<12} | {'PERCENT':<10} | {'RETAINED'}")
    print("-" * 65)

    # Generiamo soglie dal 95% in giù fino allo 0%, a passi del 5%
    # Usiamo range interi per evitare problemi di virgola mobile
    thresholds = range(95, -5, -5)

    for t_int in thresholds:
        threshold = t_int / 100.0

        # Logica Cumulativa: Conta quanti sono MAGGIORI della soglia
        count = np.sum(all_scores > threshold)
        pct = (count / total) * 100

        # Barra visuale
        bar_len = int(pct / 5)  # 1 char = 5%
        bar = "█" * bar_len

        label = f"> {threshold * 100:3.0f}%"
        print(f"{label:<15} | {count:<12} | {pct:6.2f}%    | {bar}")

    print("-" * 65)
    print(f"Total Windows: {total}")


def main():
    if not os.path.exists(HDF5_PATH):
        print(f"❌ File non trovato: {HDF5_PATH}")
        return

    dataset = MSADataset(HDF5_PATH)
    total_windows = len(dataset)

    print(f"📊 Raccolta dati da {total_windows} finestre...")

    # 1. Raccogliamo TUTTI i punteggi in un array numpy
    # (È veloce e occupa poca memoria anche per 100k elementi)
    all_scores = np.zeros(total_windows, dtype=float)

    for i, item in enumerate(tqdm(dataset, desc="Scanning")):
        all_scores[i] = calculate_block_identity(item.sequences)

    # 2. Stampa Tabella
    print_cumulative_table(all_scores)

    # 3. Grafico Cumulativo (Curva di Sopravvivenza)
    try:
        # Ordiniamo i punteggi per il plot
        sorted_scores = np.sort(all_scores)
        # Asse Y: da 1.0 (100%) a 0.0, decrescente
        yvals = np.arange(len(sorted_scores), 0, -1) / len(sorted_scores)

        plt.figure(figsize=(10, 6))
        plt.plot(sorted_scores, yvals, linewidth=2, color='darkorange')

        # Abbellimenti
        plt.fill_between(sorted_scores, yvals, color='orange', alpha=0.1)
        plt.xlabel('Minimum Identity Threshold')
        plt.ylabel('Fraction of Dataset Retained')
        plt.title('Reverse Cumulative Distribution of Identity\n(How much data do I keep if I filter at X?)')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)

        # Aggiunge linee guida per il 50% e 80% di retention
        plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='50% Retention')
        plt.legend()

        plt.savefig(PLOT_OUTPUT)
        print(f"\n✅ Grafico salvato come: {PLOT_OUTPUT}")

    except Exception as e:
        print(f"Grafico non generato: {e}")

    dataset.close()


if __name__ == "__main__":
    main()