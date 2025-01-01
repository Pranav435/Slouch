# watchdog.py
import psutil
import subprocess
import sys
import time
import os
import logging
import win32event
import win32api
import winerror

# Configuration
MAIN_APP_NAME = "slouch.exe"  # Name of the main executable
MAIN_APP_PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), MAIN_APP_NAME)
CHECK_INTERVAL = 10  # seconds between checks
MUTEX_NAME = "Global\\SlouchWatchdogMutex"

# Logging Setup
LOG_DIR = os.path.join(os.getenv('APPDATA'), 'Slouch')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FILE = os.path.join(LOG_DIR, 'watchdog_log.log')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def is_process_running(process_name):
    """Check if there is any running process that contains the given name."""
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'].lower() == process_name.lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False

def start_main_app():
    """Start the main application."""
    try:
        subprocess.Popen([MAIN_APP_PATH], shell=False)
        logging.info(f"Started main application: {MAIN_APP_PATH}")
    except Exception as e:
        logging.error(f"Failed to start main application: {e}")

def is_watchdog_already_running():
    """Check if another instance of the watchdog is already running."""
    try:
        mutex = win32event.CreateMutex(None, False, MUTEX_NAME)
        last_error = win32api.GetLastError()
        if last_error == winerror.ERROR_ALREADY_EXISTS:
            return True
        return False
    except Exception as e:
        logging.error(f"Error creating mutex: {e}")
        return False

def main():
    if is_watchdog_already_running():
        logging.info("Another instance of Watchdog is already running. Exiting.")
        sys.exit(0)
    
    logging.info("Watchdog started.")
    while True:
        if not is_process_running(MAIN_APP_NAME):
            logging.warning(f"{MAIN_APP_NAME} not running. Attempting to start.")
            start_main_app()
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    # Ensure the main application path exists
    if not os.path.exists(MAIN_APP_PATH):
        logging.error(f"Main application not found at {MAIN_APP_PATH}. Watchdog exiting.")
        sys.exit(1)
    main()
