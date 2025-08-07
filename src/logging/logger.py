import logging
import sys
from pathlib import Path

def setup_logger(name=__name__, log_file="expression_identifier.log"):
    """Configure and return a configured logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels by default

    # Prevent propagating to root logger to avoid duplicate logs
    logger.propagate = False
    
    # Create logs directory if not exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler (rotating)
    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)  # Only show INFO+ in console
    
    # Add handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# Initialize root logger
logger = setup_logger("ExpressionIdentifier")
