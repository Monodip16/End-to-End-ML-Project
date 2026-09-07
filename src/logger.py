import logging
import os
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log"
logs_path = os.path.join(os.getcwd(),"logs", LOG_FILE)

os.makedirs(logs_path, exist_ok = True)
log_file_path = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename = log_file_path,
    level = logging.INFO,
    format = "[%(asctime)s] %(levelname)s - %(message)s]",
    datefmt = "%Y-%m-%d %H:%M:%S"
)



