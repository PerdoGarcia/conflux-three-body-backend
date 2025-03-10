# fixed_senti.py
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

import os
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

        # Initialize RAKE - do this first as it's lightest
        logger.info("Initializing keyword extractor...")
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
        # Initialize RAKE
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
        # Check if models are initialized
        global sentiment_pipeline, emotion_pipeline, sbert_model, rake_extractor
        if not all([sentiment_pipeline, emotion_pipeline, sbert_model, rake_extractor]):
            logger.warning("Models not initialized, attempting to initialize now")
            initialize_models()

        # Force JSON parsing in case Content-Type is not set properly
        data = request.get_json(force=True)
        message_text = data.get('message_text', '')

        # stored_texts is expected to be a list of objects
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

        if not message_text:
            return jsonify({'error': 'Text parameter is required'}), 400

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

        logger.debug(f"Processed result for text of length {len(message_text)}")

        # Use NumpyEncoder to handle numpy types
        return json.dumps(response_obj, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        logger.error(f"Error in get_sentiment: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def get_sentiment_analysis(text):
    """Analyze sentiment using the pre-loaded pipeline"""
    global sentiment_pipeline
    try:
        result = sentiment_pipeline(text)[0]
        return {"label": result['label'], "score": float(result['score'])}
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return {"label": "neutral", "score": 0.5}

def get_emotion_analysis(text):
    """Analyze emotions using the pre-loaded pipeline"""
    global emotion_pipeline
    try:
        emotions = emotion_pipeline(text)[0]
        return {e['label']: float(e['score']) for e in emotions}
    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        return {"neutral": 1.0}

def get_intensity(emotion_scores):
    """Calculate emotional intensity from emotion scores"""
    if not emotion_scores:
        return 0.0
    return float(max(emotion_scores.values()))

def extract_keywords(text, num_keywords=5):
    """Extract keywords using pre-loaded RAKE"""
    global rake_extractor
    try:
        rake_extractor.extract_keywords_from_text(text)
        phrases = rake_extractor.get_ranked_phrases()
        return phrases[:num_keywords] if phrases else []
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []

def get_text_similarity(text1, text2):
    """Calculate text similarity using sentence embeddings"""
    global sbert_model
    try:
        embeddings = sbert_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    except Exception as e:
        logger.error(f"Error calculating text similarity: {e}")
        return 0.0

def get_closeness(text1, text2):
    """Calculate overall closeness between two texts"""
    try:
        emotion1 = get_emotion_analysis(text1)
        emotion2 = get_emotion_analysis(text2)

        emotions1_vector = list(emotion1.values())
        emotions2_vector = list(emotion2.values())
        emotion_similarity = float(cosine_similarity([emotions1_vector], [emotions2_vector])[0][0])

        text_similarity = get_text_similarity(text1, text2)

        intensity1 = get_intensity(emotion1)
        intensity2 = get_intensity(emotion2)
        intensity_similarity = 1.0 - abs(intensity1 - intensity2)

        closeness_score = (emotion_similarity + text_similarity + intensity_similarity) / 3.0
        return {
            "closeness_score": float(closeness_score),
            "emotion_similarity": float(emotion_similarity),
            "text_similarity": float(text_similarity),
            "intensity_similarity": float(intensity_similarity)
        }
    except Exception as e:
        logger.error(f"Error calculating closeness: {e}")
        return {
            "closeness_score": 0.0,
            "emotion_similarity": 0.0,
            "text_similarity": 0.0,
            "intensity_similarity": 0.0
        }

def process_input(text, stored_inputs):
    """Process a single text input and return analysis results"""
    logger.info(f"Processing input text of length: {len(text)}")

    sentiment = get_sentiment_analysis(text)
    emotion = get_emotion_analysis(text)
    intensity = get_intensity(emotion)
    keywords = extract_keywords(text)

    result = {
        "sentiment": sentiment,
        "emotion": emotion,
        "intensity": intensity,
        "keywords": keywords
    }

    input_data = {
        "id": len(stored_inputs) + 1,
        "text": text,
    }
    stored_inputs.append(input_data)

    if len(stored_inputs) > 1:
        highest_closeness = -1.0
        highest_closeness_id = None
        highest_closeness_data = None

        for previous_input in stored_inputs[:-1]:
            # Safely get the text from the previous input
            prev_text = previous_input.get('text', '') if isinstance(previous_input, dict) else str(previous_input)

            try:
                # Get closeness data
                closeness_data = get_closeness(prev_text, text)
                closeness_score = closeness_data["closeness_score"]

                if closeness_score > highest_closeness:
                    highest_closeness = closeness_score
                    highest_closeness_id = previous_input.get('id') if isinstance(previous_input, dict) else None
                    highest_closeness_data = closeness_data
            except Exception as e:
                logger.error(f"Error calculating closeness: {e}")
                continue

        if highest_closeness_data:
            result["highest_closeness_id"] = highest_closeness_id
            result["closeness_score"] = highest_closeness_data["closeness_score"]
            result["emotion_similarity"] = highest_closeness_data["emotion_similarity"]
            result["text_similarity"] = highest_closeness_data["text_similarity"]
            result["intensity_similarity"] = highest_closeness_data["intensity_similarity"]

    return result
