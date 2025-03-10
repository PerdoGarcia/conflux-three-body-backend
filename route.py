import os
from flask_cors import CORS
from flask import Flask
import logging

# Configure logging only once
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import your blueprint after logging is set up
try:
    from fixed_senti import sentiment_bp, initialize_models
    logger.info("Successfully imported from fixed_senti module")
except ImportError as e:
    logger.error(f"Error importing from fixed_senti: {e}")
    raise

def create_app():
    app = Flask(__name__)
    
    # Initialize models and resources
    with app.app_context():
        try:
            # Initialize models (only happens once)
            # NLTK resources should already be downloaded by nltk_setup.py
            initialize_models()
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    app.register_blueprint(sentiment_bp)
    
    CORS(app, origins=["http://localhost:3000", "http://localhost:5500",
                     "http://127.0.0.1:3000", "http://127.0.0.1:5500",
                     "http://localhost", "http://127.0.0.1",
                     "https://conflux-5ac7b.web.app", "http://192.168.0.10"],
         supports_credentials=True)
    
    return app

# Create the app instance for gunicorn
app = create_app()

# Only used when running directly, not via gunicorn
if __name__ == '__main__':
    try:
        print("Starting application and loading AI models (this may take a few minutes)...")
        port = int(os.getenv('PORT', 8000))
        logger.info(f"Server is now ready! Running on port {port}")
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
