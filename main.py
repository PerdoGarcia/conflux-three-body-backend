import os
from flask_cors import CORS
from flask import Flask
import nltk
import logging
import torch
from routes.fixed_senti import sentiment_bp, initialize_models

# Configure logging only once
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)

    # Initialize models and resources
    try:
        # Download NLTK resources
        nltk.download('punkt')
        initialize_models()
        logger.info("Models and resources initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models or resources: {e}")
        raise

    app.register_blueprint(sentiment_bp)
    CORS(app, origins=["http://localhost:3000", "http://localhost:5500",
                    "http://127.0.0.1:3000", "http://127.0.0.1:5500",
                    "http://localhost", "http://127.0.0.1",
                    "https://conflux-5ac7b.web.app", "http://192.168.0.10"],
         supports_credentials=True)
    return app

# Create the application instance - this is what Gunicorn will look for
app = create_app()

# For development server only
if __name__ == '__main__':
    try:
        port = int(os.getenv('PORT', 8080))
        logger.info(f"Development server is running on port {port}")
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        logger.error(f"Failed to start development server: {e}")