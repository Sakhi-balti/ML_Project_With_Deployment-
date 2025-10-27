import os
import logging
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_PATH = os.path.join(os.getcwd(), "logs",LOG_FILE)
LOG_FILE_PATH = os.path.join(LOG_PATH, LOG_FILE)

os.makedirs(LOG_PATH, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
    level=logging.INFO
)
