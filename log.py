from datetime import datetime
from config import config

# get log file path from config
LOG_FILE = config["LOG_FILE"]      

def log_plate(plate, speed):           
    # get timestamp
    timemstamp = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    # open log file in append mode
    with open(LOG_FILE, "a") as f:
        # write the log
        f.write(f"{plate} - {int(speed)} km/h - {timemstamp}\n")