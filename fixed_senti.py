import json
import sys
import nltk
import time
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rake_nltk import Rake
import os
import logging
import boto3
import tempfile


# Setup logging
logger = logging.getLogger(__name__)

# Create the Blueprint
sentiment_bp = Blueprint('sentiment', __name__)

# Global variables to store models and tokenizers
sentiment_pipeline = None
emotion_pipeline = None
sbert_model = None
rake_extractor = None
stored_texts = []

# Configure NLTK to use local data directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_dir)

def get_s3_client():
    """Create and return an S3 client"""
    return boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )

def download_from_s3(bucket, s3_prefix, local_directory):
    """Download files from S3 to a local directory"""
    logger.info(f"Downloading model from S3: {s3_prefix}")
    s3 = get_s3_client()
    bucket_name = os.environ.get('S3_BUCKET_NAME', bucket)

    # Create local directory if it doesn't exist
    os.makedirs(local_directory, exist_ok=True)

    # List objects in the S3 prefix
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

    for page in pages:
        if 'Contents' not in page:
            continue

        for obj in page['Contents']:
            # Get the relative path
            key = obj['Key']
            # Skip if it's the directory itself
            if key.endswith('/'):
                continue

            # Determine local file path
            relative_path = key[len(s3_prefix):].lstrip('/')
            local_file_path = os.path.join(local_directory, relative_path)

            # Create directories if needed
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the file
            logger.info(f"Downloading {key} to {local_file_path}")
            s3.download_file(bucket_name, key, local_file_path)

    logger.info(f"Downloaded model from S3 to {local_directory}")

def load_sentiment_model_from_s3():
    """Load sentiment model from S3"""
    logger.info("Loading sentiment model from S3...")
    try:
        model_dir = tempfile.mkdtemp()
        tokenizer_dir = tempfile.mkdtemp()

        # Download from S3 - using exact folder names from your S3 bucket
        download_from_s3(
            os.environ.get('S3_BUCKET_NAME', 'conflux-three-body'),
            'sentiment_model/',
            model_dir
        )
        download_from_s3(
            os.environ.get('S3_BUCKET_NAME', 'conflux-three-body'),
            'sentiment_tokenizer/',
            tokenizer_dir
        )

        # Load models
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        sentiment_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        logger.info("Sentiment model loaded successfully from S3")
        return sentiment_model, sentiment_tokenizer
    except Exception as e:
        logger.error(f"Error loading sentiment model from S3: {e}")
        raise

def load_emotion_model_from_s3():
    """Load emotion model from S3"""
    logger.info("Loading emotion model from S3...")
    try:
        model_dir = tempfile.mkdtemp()
        tokenizer_dir = tempfile.mkdtemp()

        # Download from S3 - using exact folder names from your S3 bucket
        download_from_s3(
            os.environ.get('S3_BUCKET_NAME', 'conflux-three-body'),
            'emotion_model/',
            model_dir
        )
        download_from_s3(
            os.environ.get('S3_BUCKET_NAME', 'conflux-three-body'),
            'emotion_tokenizer/',
            tokenizer_dir
        )

        # Load models
        emotion_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        emotion_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)

        logger.info("Emotion model loaded successfully from S3")
        return emotion_model, emotion_tokenizer
    except Exception as e:
        logger.error(f"Error loading emotion model from S3: {e}")
        raise

def load_sbert_model_from_s3():
    """Load SBERT model from S3"""
    logger.info("Loading SBERT model from S3...")
    try:
        model_dir = tempfile.mkdtemp()

        # Download from S3 - using exact folder name from your S3 bucket
        download_from_s3(
            os.environ.get('S3_BUCKET_NAME', 'conflux-three-body'),
            'sbert/',
            model_dir
        )

        # Load model
        sbert_model = SentenceTransformer(model_dir)

        logger.info("SBERT model loaded successfully from S3")
        return sbert_model
    except Exception as e:
        logger.error(f"Error loading SBERT model from S3: {e}")
        raise

def initialize_models():
    """Initialize all models from S3"""
    global sentiment_pipeline, emotion_pipeline, sbert_model, rake_extractor

    logger.info("Loading NLP models from S3...")

    try:
        # Check if AWS environment variables are set
        if not all([
            os.environ.get('AWS_ACCESS_KEY_ID'),
            os.environ.get('AWS_SECRET_ACCESS_KEY'),
            os.environ.get('S3_BUCKET_NAME')
        ]):
            logger.warning("AWS credentials not found in environment variables. Using fallback initialization.")
            return initialize_models_fallback()

        # Initialize RAKE with English stopwords only
        logger.info("Initializing keyword extractor with English stopwords...")
        try:
            from nltk.corpus import stopwords
            english_stopwords = stopwords.words('english')
            rake_extractor = Rake(stopwords=english_stopwords)
            logger.info("Initialized RAKE with English stopwords")
        except Exception as e:
            logger.warning(f"Could not initialize RAKE with English stopwords: {e}. Using default.")
            rake_extractor = Rake()

        # Load models from S3
        try:
            # Load sentiment model
            sentiment_model, sentiment_tokenizer = load_sentiment_model_from_s3()

            # Load emotion model
            emotion_model, emotion_tokenizer = load_emotion_model_from_s3()

            # Load SBERT model
            sbert_model = load_sbert_model_from_s3()

            # Initialize pipelines
            logger.info("Setting up inference pipelines...")
            device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
            logger.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")

            sentiment_pipeline = pipeline(
                "text-classification",
                model=sentiment_model,
                tokenizer=sentiment_tokenizer,
                device=device
            )
            emotion_pipeline = pipeline(
                "text-classification",
                model=emotion_model,
                tokenizer=emotion_tokenizer,
                top_k=3,
                device=device
            )

            logger.info("All models loaded successfully from S3!")
            return True

        except Exception as e:
            logger.error(f"Error loading models from S3: {e}")
            return initialize_models_fallback()

    except Exception as e:
        logger.error(f"Error in model initialization: {e}")
        return initialize_models_fallback()

def initialize_models_fallback():
    """Fallback method to initialize models from HuggingFace directly"""
    global sentiment_pipeline, emotion_pipeline, sbert_model, rake_extractor

    logger.warning("Using fallback method to load models directly from HuggingFace")

    try:
        # Initialize RAKE with English stopwords
        try:
            from nltk.corpus import stopwords
            english_stopwords = stopwords.words('english')
            rake_extractor = Rake(stopwords=english_stopwords)
        except:
            rake_extractor = Rake()

        # Use smaller models or direct loading
        sentiment_pipeline = pipeline("sentiment-analysis")
        emotion_pipeline = pipeline(
            "text-classification",
            model="joeddav/distilbert-base-uncased-go-emotions-student",
            top_k=3
        )
        sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # Smaller model

        logger.info("Fallback initialization successful")
        return True
    except Exception as fallback_error:
        logger.error(f"Fallback initialization failed: {fallback_error}")
        raise

class NumpyEncoder(json.JSONEncoder):
    """Special JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.void):
            return None
        return super(NumpyEncoder, self).default(obj)
@sentiment_bp.route('/analyze', methods=['GET', 'POST'])
def get_sentiment():
    """API endpoint to analyze text sentiment and emotions"""
    try:
        # Log the incoming request for debugging
        logger.info(f"Received request: Method={request.method}, Content-Type={request.headers.get('Content-Type', 'None')}")
        request_data = request.get_data()
        logger.info(f"Request data (first 200 chars): {request_data[:200]}")

        # Check if models are initialized
        global sentiment_pipeline, emotion_pipeline, sbert_model, rake_extractor
        if not all([sentiment_pipeline, emotion_pipeline, sbert_model, rake_extractor]):
            logger.warning("Models not initialized, attempting to initialize now")
            initialize_models()

        # More robust JSON parsing with better error handling
        try:
            # Try to get JSON data with explicit Content-Type check first
            if request.is_json:
                data = request.get_json()
                logger.info("Successfully parsed JSON from request with Content-Type: application/json")
            else:
                # As a fallback, try to parse it anyway, but log the attempt
                logger.warning("Request does not have Content-Type: application/json, attempting to parse anyway")
                data = request.get_json(force=True)
                logger.info("Successfully parsed JSON despite missing Content-Type header")
        except Exception as json_error:
            logger.error(f"JSON parsing error: {json_error}")
            # Provide a default empty dict if JSON parsing fails
            data = {}

        # Log the parsed data for debugging
        logger.info(f"Parsed data: {data}")

        # Get required fields with better validation
        message_text = data.get('message_text', '')
        if not message_text:
            logger.warning("Missing required 'message_text' parameter")
            if request.args.get('message_text'):  # Check if it's in URL params instead
                message_text = request.args.get('message_text')
                logger.info(f"Found message_text in URL parameters: {message_text[:50]}...")
            else:
                return jsonify({'error': 'Text parameter is required'}), 400

        # Rest of your code remains the same
        stored_texts = data.get('stored_texts', [])

        # Validate stored_texts format
        validated_stored_texts = []
        for item in stored_texts:
            if isinstance(item, dict) and 'text' in item and 'id' in item:
                validated_stored_texts.append(item)
            elif isinstance(item, dict) and 'id' in item:
                validated_stored_texts.append({'id': item['id'], 'text': str(item.get('message', ''))})
            else:
                # Create a proper dict if missing required fields
                validated_stored_texts.append({'id': len(validated_stored_texts) + 1, 'text': str(item)})

        # Process the input using pre-loaded models
        result = process_input(message_text, validated_stored_texts)

        # Get the new message (it will be the last entry in stored_texts)
        new_message = validated_stored_texts[-1] if validated_stored_texts else {"id": 1, "text": message_text}

        # Build the final response object
        response_obj = {
            "emotion": result.get("emotion"),
            "intensity": float(result.get("intensity", 0)),
            "keywords": result.get("keywords"),
            "messageText": message_text,
            "sentiment": result.get("sentiment"),
            "id": new_message.get("id"),
            "text": new_message.get("text"),
            "time": time.time(),
            "highest_closeness_id": result.get("highest_closeness_id"),
            "closeness_score": float(result.get("closeness_score", 0)) if result.get("closeness_score") is not None else None,
            "emotion_similarity": float(result.get("emotion_similarity", 0)) if result.get("emotion_similarity") is not None else None,
            "text_similarity": float(result.get("text_similarity", 0)) if result.get("text_similarity") is not None else None,
            "intensity_similarity": float(result.get("intensity_similarity", 0)) if result.get("intensity_similarity") is not None else None
        }

        logger.info(f"Successfully processed text. Response length: {len(json.dumps(response_obj))}")

        # Use NumpyEncoder to handle numpy types
        return json.dumps(response_obj, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        logger.error(f"Error in get_sentiment: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500