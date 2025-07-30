#!/usr/bin/env python3
"""
Logging Configuration for Project GUI

Usage:
    from logger_config import get_logger
    logger = get_logger(__name__)
    logger.info("Application started")
    logger.error("Model loading failed", exc_info=True)

Date: 2025-07-29
Version: 1.0
"""

import logging
import logging.handlers
import os
from pathlib import Path

def setup_logging(log_level=logging.INFO, log_dir="logs"):
    """
    Configure application-wide logging.
    
    Creates rotating log files to prevent disk space issues and provides
    structured logging for both real-time monitoring and historical analysis.
    
    Args:
        log_level: Minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files (created if doesn't exist)
    
    Returns:
        logging.Logger: Configured root logger
    """
    
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create formatter for structured log entries
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler for real-time monitoring
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Show INFO and above on console
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Rotating file handler for persistent storage
    # Rotates at 10MB, keeps 5 backup files (50MB total max)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, 'system.log'),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)  # Log everything to file
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Error-only file for critical issues
    error_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, 'error.log'),
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    return root_logger

def get_logger(name):
    """
    Get a logger for a specific module.
    
    Args:
        name (str): Usually __name__ to identify the calling module
        
    Returns:
        logging.Logger: Module-specific logger
    """
    return logging.getLogger(name)

# Initialize logging when module is imported
setup_logging()