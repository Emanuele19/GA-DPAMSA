import requests, time
import sys
import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path

BASE = "https://data.orthodb.org/v12/download/odb_data_dump"
API = "https://data.orthodb.org/v12/fasta"

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "dumps"
DATASET_ROOT = SCRIPT_DIR / "../fasta_files"

# Cartella di destinazione rigorosamente separata
INFERENCE_RAW_DIR = DATASET_ROOT / "orthodb_v12/inference_raw"

TRAIN_OFFSET = 20000      # I dati che NON dobbiamo toccare
MAX_DOWNLOAD = 100        # Quanti file di test vogliamo

CLADE = "Mammalia"
OGS_CACHE = Path("ogs.txt")

def download_cds_for_og(og_id):
    url = f"{API}?id={og_id}&seqtype=cds"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    (INFERENCE_RAW_DIR / f"{og_id}.cds.fasta").write_text(r.text, encoding="utf-8")

def main():
    INFERENCE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Leggi il file cache degli OGs
    if not OGS_CACHE.exists() or OGS_CACHE.stat().st_size == 0:
        print("Errore: ogs.txt non trovato. Assicurati di aver scaricato i dati base.")
        sys.exit(1)
        
    ogs = [line.strip() for line in OGS_CACHE.read_text().splitlines() if line.strip()]
    
    # 2. SLICE DI SICUREZZA: partiamo dal 20.000 in poi
    ogs_inference = ogs[TRAIN_OFFSET:]
    total_iter = min(MAX_DOWNLOAD, len(ogs_inference))
    
    print(f"Inizio download di {total_iter} file vergini per l'Inferenza...")
    
    for i, og in enumerate(tqdm(ogs_inference[:total_iter], desc="Download Inferenza"), 1):
        if not og: continue
        out = INFERENCE_RAW_DIR / f"{og}.cds.fasta"
        if out.exists() and out.stat().st_size > 0:
            continue
            
        try:
            download_cds_for_og(og)
        except Exception as e:
            tqdm.write(f"Fallito {og}: {e}")

        if i % 50 == 0:
            time.sleep(1)

    print(f"\nDownload completato in: {INFERENCE_RAW_DIR}")

if __name__ == "__main__":
    main()