import os
import logging
import subprocess

# Add this near the start of your app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log file structure info
def log_file_structure():
    logger.info("Checking NLTK data directories...")

    # Check environment variable
    logger.info(f"NLTK_DATA env var: {os.environ.get('NLTK_DATA')}")

    # List contents of potential NLTK directories
    dirs_to_check = [
        '/app/nltk_data',
        '/opt/venv/nltk_data',
        '/root/nltk_data',
        '/usr/local/share/nltk_data'
    ]

    for directory in dirs_to_check:
        if os.path.exists(directory):
            logger.info(f"Directory exists: {directory}")
            try:
                # Run ls command and capture output
                result = subprocess.run(
                    ['find', directory, '-type', 'd', '-maxdepth', '3'],
                    capture_output=True, text=True, check=True
                )
                logger.info(f"Contents:\n{result.stdout}")
            except Exception as e:
                logger.info(f"Error listing directory: {e}")
        else:
            logger.info(f"Directory does not exist: {directory}")

# Call this at application startup
log_file_structure()