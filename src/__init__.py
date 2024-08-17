import os 
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir='logs'
log_filepath=os.path.join(log_dir, 'running_logs.log')
os.makedirs(log_dir, exist_ok=True)#no error will be thrown if directory exists

logging.basicConfig(
    level=logging.INFO,#logs messages at level info and higher
    format=logging_str,#format as defined earlier
    handlers=[
        logging.FileHandler(log_filepath),#in the file
        logging.StreamHandler(sys.stdout)#at the terminal
    ]
)

logger=logging.getLogger('MyLogger')#creates a logger of name MyLogger