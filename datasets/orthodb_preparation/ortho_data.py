import requests, time
import sys
import os
from tqdm import tqdm
import pandas as pd
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from dataset_module.data_utils import *

BASE = "https://data.orthodb.org/v12/download/odb_data_dump"

DATASET_ROOT= Path("../fasta_files")
OUT = Path("dumps")

CLADE = "Mammalia"  # Vertebrate/Mammalia/Eukaryota/…

API = "https://data.orthodb.org/v12/fasta"
raw_dir = DATASET_ROOT / "orthodb_v12/raw"; raw_dir.mkdir(parents=True, exist_ok=True)

# if not none, max number of files downloaded
max_download_count = 500

def download_files():
    OUT.mkdir(parents=True, exist_ok=True)

    files = [
        "odb12v2_levels.tab.gz",
        "odb12v2_level2species.tab.gz",
        "odb12v2_OG2genes.tab.gz",
        "odb12v2_genes.tab.gz",
        "odb12v2_OGs.tab.gz",
    ]

    for fn in files:
        url = f"{BASE}/{fn}"
        dst = OUT / fn
        if dst.exists() and dst.stat().st_size > 0:
            continue

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dst, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=fn
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

def get_ogs_for_clade():
    levels = pd.read_csv(
        OUT / "odb12v2_levels.tab.gz",
        sep="\t", header=None, compression="gzip",
        names=["level_taxid", "name", "nr_genes", "nr_ogs", "nr_species"],
    )

    level_id = levels.loc[levels["name"] == CLADE, "level_taxid"].iloc[0]
    print("Clade:", CLADE, "→ level_taxid:", level_id)

    ogs_df = pd.read_csv(
        OUT / "odb12v2_OGs.tab.gz",
        sep="\t", header=None, compression="gzip",
        names=["og_id", "level_taxid", "og_name"],
    )

    ogs_in_clade = ogs_df.loc[ogs_df["level_taxid"]==level_id, "og_id"].drop_duplicates()
    print("OG trovati:", len(ogs_in_clade))
    return ogs_in_clade.tolist()

def download_cds_for_og(og_id):
    url = f"{API}?id={og_id}&seqtype=cds"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    (raw_dir/f"{og_id}.cds.fasta").write_text(r.text, encoding="utf-8")


def download_all_fastas():
    raw_dir.mkdir(parents=True, exist_ok=True)

    ogs = get_ogs_for_clade()
    total_iter = min(max_download_count, len(ogs)) if max_download_count else len(ogs)

    for i, og in enumerate(tqdm(ogs[:total_iter], total=min(total_iter, max_download_count), desc="Scarico OG", unit="og", file=sys.stdout), 1):
        if not og: continue
        out = raw_dir/f"{og}.cds.fasta"
        if out.exists() and out.stat().st_size>0:
            continue
        try:
            download_cds_for_og(og)
        except Exception as e:
            tqdm.write(f"Failed {i}: {e}")

        if i % 50 == 0:
            time.sleep(1)
        tqdm.write(f"Downloaded {i} {og}")

def main():
    download_files()
    download_all_fastas()


if __name__ == "__main__":
    main()