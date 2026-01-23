import logging
import os
from datetime import datetime

def setup_logger(log_dir: str = None, log_console: bool = True):
    """Configura un logger che scrive sia su file che su console."""
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Nome file basato sul timestamp per non sovrascrivere sessioni diverse
    log_filename = datetime.now().strftime("training_%Y%m%d_%H%M%S.log")
    
    # Creazione del logger
    logger = logging.getLogger("MSA_Trainer")
    logger.setLevel(logging.INFO)
    
    # Formattatore: [Data Ora] [Livello] Messaggio
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Handler per Console
    if log_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Handler per File
    if log_dir:
        log_path = os.path.join(log_dir, log_filename)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler) 
    
    return logger