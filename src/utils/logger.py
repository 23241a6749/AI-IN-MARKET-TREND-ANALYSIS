"""
Comprehensive logging system for the Multimodal Market Intelligence System.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "market_intelligence",
    log_dir: str = "logs",
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up a comprehensive logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Daily log file
        log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    return logger


def log_model_training(logger: logging.Logger, epoch: int, train_loss: float, 
                       train_acc: float, val_loss: float, val_acc: float):
    """Log training metrics for an epoch."""
    logger.info(
        f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )


def log_data_collection(logger: logging.Logger, source: str, records: int, 
                       date_range: tuple):
    """Log data collection results."""
    logger.info(
        f"Collected {records} {source} records from {date_range[0]} to {date_range[1]}"
    )


def log_model_evaluation(logger: logging.Logger, model_name: str, metrics: dict):
    """Log model evaluation metrics."""
    logger.info(f"Model: {model_name}")
    logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
    logger.info(f"  Recall: {metrics.get('recall', 0):.4f}")
    logger.info(f"  F1 Score: {metrics.get('f1_score', 0):.4f}")

