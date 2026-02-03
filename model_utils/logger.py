import logging
import os
from datetime import datetime

def setup_logger(log_dir: str = None, log_console: bool = True):
    """Configura un logger robusto per training di RL."""
    logger = logging.getLogger("MSA_Trainer")
    
    # Se il logger ha già degli handler, non aggiungerne altri
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    # Evita che i log salgano al logger root di sistema
    logger.propagate = False 
    
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler Console
    if log_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Handler File
    if log_dir:
        os.makedirs(log_dir, exist_ok=True) # Più compatto di if not os.path.exists
        log_filename = datetime.now().strftime("training_%Y%m%d_%H%M%S.log")
        log_path = os.path.join(log_dir, log_filename)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler) 
    
    return logger