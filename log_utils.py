import logging
import sys

def setup_logging(level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)

    # Handler console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
