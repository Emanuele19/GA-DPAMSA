# ---- Base (CPU) --------------------------------------------------------------
FROM python:3.10-slim AS base
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Evita interazioni nei comandi APT + set Python
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# Dipendenze di sistema minime (numpy/pandas/matplotlib, curl, git, xz)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    xz-utils bzip2 python3-tk\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia solo i requirements per sfruttare la cache
COPY requirements.txt ./requirements.txt

# Separa torch dal resto per gestire CPU/GPU
RUN awk '!/^torch==/' requirements.txt > requirements.no-torch.txt

# Aggiorna strumenti python e installa deps (senza torch)
RUN pip install --upgrade pip==25.0.1 wheel==0.41.2 setuptools==68.2.0 && \
    pip install --no-cache-dir -r requirements.no-torch.txt

# Scegli il canale torch a build-time: cpu (default) o cu121
ARG PYTORCH_CHANNEL=cpu

# Installa torch 2.2.1 dal canale adatto
RUN if [ "$PYTORCH_CHANNEL" = "cpu" ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.1 ; \
    else \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.2.1 ; \
    fi

# Cartella per output (montabile come volume)
RUN mkdir -p /app/result

# Test veloce ambiente
RUN python - <<'PY'
import torch, numpy, pandas, matplotlib
print('Torch:', torch.__version__)
print('NumPy:', numpy.__version__)
print('Pandas:', pandas.__version__)
print('Matplotlib backend:', matplotlib.get_backend())
PY

# ---------------- micromamba + tool bio (NO muscle5) --------------------------
# Root micromamba + PATH
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=$PATH:/opt/micromamba/bin

# --- requisiti per estrarre .tar.bz2
# RUN apt-get update && apt-get install -y --no-install-recommends bzip2 && rm -rf /var/lib/apt/lists/*

# --- installa micromamba
RUN set -eux; \
    mkdir -p /tmp/mamba; \
    curl -fL https://micro.mamba.pm/api/micromamba/linux-64/latest -o /tmp/micromamba.tar.bz2; \
    tar -xj -C /tmp/mamba -f /tmp/micromamba.tar.bz2 bin/micromamba; \
    install -m 0755 /tmp/mamba/bin/micromamba /usr/local/bin/micromamba; \
    rm -rf /tmp/mamba /tmp/micromamba.tar.bz2


# Crea l'ambiente "msatools" con i tool da docs (senza MUSCLE5, da installare a parte)
# Tool: clustalo, clustalw, mafft, msaprobs, hmmer, sepp, pasta
RUN micromamba create -y -n msatools -c conda-forge -c bioconda \
        clustalo clustalw mafft msaprobs hmmer sepp pasta && \
    micromamba clean -y --all

# Aggiungi l'ambiente al PATH (senza "attivazione" interattiva)
ENV PATH=$PATH:/opt/micromamba/envs/msatools/bin
ENV CONDA_PREFIX=/opt/micromamba/envs/msatools

# Verifiche rapide (non fallire se un tool stampa su stderr con -h)
RUN which clustalo && clustalo --version || true; \
    which clustalw && clustalw -help | head -n 1 || true; \
    which mafft && mafft --version || true; \
    which msaprobs && msaprobs -h | head -n 1 || true; \
    which hmmsearch && hmmsearch -h | head -n 3 || true; \
    which run_sepp.py || true; \
    which run_pasta.py || true

# ---- MUSCLE5 (installazione da binario ufficiale) ----------------------------
ARG MUSCLE_URL=https://github.com/rcedgar/muscle/releases/download/v5.3/muscle-linux-x86.v5.3
ARG MUSCLE_SHA256=318abeb951d786a3e2532714cc81ad3b3d8f79a2b517dc31316eeb5b694db2bc

RUN set -euo pipefail && \
    TMPD="$(mktemp -d)" && cd "$TMPD" && \
    echo "Scarico $MUSCLE_URL" && \
    curl -fL --retry 3 -o muscle.pkg "$MUSCLE_URL" && \
    echo "${MUSCLE_SHA256}  muscle.pkg" | sha256sum -c - && \
    if file muscle.pkg | grep -qi 'gzip compressed'; then \
        tar -xzf muscle.pkg; \
      elif file muscle.pkg | grep -qi 'xz compressed'; then \
        unxz -c muscle.pkg > muscle5 || tar -xJf muscle.pkg; \
      fi && \
    CANDIDATE="$(find . -maxdepth 3 -type f \( -name 'muscle5*' -o -name 'muscle*' \) -perm /u+x | head -n1 || true)" && \
    if [ -z "$CANDIDATE" ]; then chmod +x muscle.pkg || true; CANDIDATE="./muscle.pkg"; fi && \
    install -m 0755 "$CANDIDATE" /usr/local/bin/muscle5 && \
    ln -sf /usr/local/bin/muscle5 /usr/local/bin/muscle && \
    which muscle5 && muscle5 -h | head -n 1 && \
    rm -rf "$TMPD"


# Copia il resto del progetto
COPY . .

# Default: shell
CMD ["bash"]
