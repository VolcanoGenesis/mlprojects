import logging 
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d')}.log"
logs_path = os.path.join(os.getcwd(), 'logs', LOG_FILE)
os.makedirs(os.path.dirname(logs_path), exist_ok=True)

LOG_FILE_PATH = logs_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()  # This adds console output
    ]
)
