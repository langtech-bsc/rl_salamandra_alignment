"""Top-level package for RL - Salamandra Alignment."""

__author__ = """LangTech BSC"""
__email__ = 'langtech@bsc.es'
__version__ = '0.1.0'

import logging

# Create the logger
logger = logging.getLogger("RL_salamandra_alignment")

def setup_logging(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
    """
    Set up logging for this package.
    Ensures proper handler setup and allows dynamic level changes.
    """
    logger.setLevel(level)  # Ensure the logger captures messages at the desired level

    # Remove existing handlers to prevent duplicates
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    # Create a new StreamHandler
    handler = logging.StreamHandler()
    handler.setLevel(level)  # Ensure the handler logs at the correct level
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)

    logger.addHandler(handler)  # Add the handler

# Ensure default logging is INFO
setup_logging()