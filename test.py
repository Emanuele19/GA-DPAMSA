import h5py
import numpy as np
import config

# Sostituisci con il percorso vero del tuo file .h5
H5_PATH = "datasets/fasta_files/orthodb_v12/hdf5_3x30.h5"


def inspect_hdf5():
    with h5py.File(H5_PATH, 'r') as f:
        dset = f['alignments']
        print(f"Shape del dataset: {dset.shape}")

        # Prendiamo la prima matrice di allineamento
        first_item = dset[0]
        print("\n--- CONTENUTO GREZZO (NumPy) ---")
        print(first_item)

        # Analizziamo i numeri presenti
        unique_numbers = np.unique(first_item)
        print(f"\nNumeri trovati nel file: {unique_numbers}")

        print("\n--- INTERPRETAZIONE CON IL TUO CONFIG ATTUALE ---")
        # Invertiamo la mappa del config
        rev_map = {v: k for k, v in config.NUCLEOTIDES_MAP.items()}
        print(f"La tua mappa inversa: {rev_map}")

        print("\nDecodifica della prima riga:")
        first_row = first_item[0]
        decoded_str = "".join([rev_map.get(x, '?') for x in first_row])
        print(decoded_str)

        if 0 in unique_numbers:
            print("\n!!! ATTENZIONE !!!")
            print("Il numero 0 è presente nel file.")
            if rev_map.get(0) == 'P':
                print("Il tuo config dice che 0 è PADDING (P).")
                print(
                    "Se questa sequenza doveva essere DNA, allora 0 era probabilmente ADENINA (A) quando il file è stato creato.")
            elif rev_map.get(0) == 'A':
                print("Il tuo config dice che 0 è ADENINA (A).")
                print("In questo caso è tutto corretto, devi solo aggiornare la debug_map del Trainer.")


if __name__ == "__main__":
    inspect_hdf5()