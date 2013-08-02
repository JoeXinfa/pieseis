import logging
import config

logger = logging.getLogger(config.NAME)
file_handler = logging.FileHandler(config.LOG_FILE)

logger.addHandler(file_handler)

if config.DEBUG:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.WARNING)
