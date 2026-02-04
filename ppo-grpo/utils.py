import logging
import sys
import os


def setup_logger(name: str = "DPAMSA", log_file: str = None, level=logging.INFO):
    """
    Configures and returns a logger instance.

    Args:
        name (str): The name of the logger (default: "DPAMSA").
        log_file (str, optional): Path to a file where logs should be saved.
                                  If None, logs are only printed to console.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: The configured logger.
    """
    # 1. Create Logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if the logger is already configured
    # (Prevents duplicate log messages if setup_logger is called twice)
    if logger.hasHandlers():
        return logger

    # 2. Define Format
    # Format: [Time] [Level] Message
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 3. Console Handler (Standard Output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 4. File Handler (Optional)
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid double logging
    logger.propagate = False

    return logger