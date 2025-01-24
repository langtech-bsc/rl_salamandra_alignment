"""Top-level package for RL - Salamandra Alignment."""

__author__ = """LangTech BSC"""
__email__ = 'langtech@bsc.es'
__version__ = '0.1.0'

import logging

# Create a logger for this package
logger = logging.getLogger("RL_salamandra_alignment")  # Root logger for the package
logger.setLevel(logging.INFO)

# Add a NullHandler to prevent "No handlers could be found" warnings
if not logger.hasHandlers():
    logger.addHandler(logging.NullHandler())


def setup_logging(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
    """
    Set up logging for this package.
    """
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)

    logger.setLevel(level)
    if not logger.hasHandlers():
        logger.addHandler(handler)


from .rl_salamandra_alignment import my_test