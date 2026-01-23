import shutil
from pathlib import Path
from typing import List, Any

class BasicCompose:
    """
    Composes several offline (file-based) transforms together.
    
    Unlike standard Compose which works on list of records in memory,
    this class manages input/output directories and temporary intermediate folders.

    Args:
        transforms (list): List of offline transform objects (classes with __call__(input_dir, output_dir)).
    """
    def __init__(self, transforms: List[Any]):
        self.transforms = transforms

    def __call__(self, input_dir: Path, output_dir: Path, force: bool = False):
        """
        Executes the pipeline.
        
        Args:
            input_dir (Path): The starting directory containing raw files.
            output_dir (Path): The final destination directory.
            force (bool): If True, runs the pipeline even if output_dir is not empty.
        """
        # 1. Controllo Cache: Se l'output esiste già, saltiamo tutto (a meno di force=True)
        if output_dir.exists() and any(output_dir.iterdir()) and not force:
            print(f"Pipeline output already exists at: {output_dir}")
            print("Skipping offline preparation.")
            return

        print(f"\n[OfflineCompose] Starting Pipeline: {input_dir.name} -> {output_dir.name}")
        
        # 2. Preparazione cartella temporanea per i passaggi intermedi
        # La creiamo allo stesso livello della cartella di output per comodità
        tmp_root = output_dir.parent / ".pipeline_tmp"
        
        # Pulizia preventiva se esisteva una vecchia temp run interrotta
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)

        current_input = input_dir
        
        try:
            # 3. Ciclo sulle trasformazioni
            for i, transform in enumerate(self.transforms):
                step_name = transform.__class__.__name__
                is_last_step = (i == len(self.transforms) - 1)
                
                # Determina dove scrivere l'output di questo passaggio
                if is_last_step:
                    step_output = output_dir
                else:
                    step_output = tmp_root / f"step_{i}_{step_name}"
                
                print(f"├── Step {i+1}/{len(self.transforms)}: {step_name}")
                
                # Esegue la trasformazione
                transform(current_input, step_output)
                
                # L'output di questo step diventa l'input del prossimo
                current_input = step_output

            print(f"[OfflineCompose] Pipeline Completed Successfully.\n")

        finally:
            # 4. Pulizia Finale: Rimuove le cartelle temporanee intermedie
            if tmp_root.exists():
                shutil.rmtree(tmp_root)