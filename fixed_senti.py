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
    print("HERE", bucket_name, os.environ.get('AWS_ACCESS_KEY_ID'), os.environ.get('AWS_SECRET_ACCESS_KEY'))
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
    except Exception as e:
        logger.error(f"Error initializing models: {e}")

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
        return {"label": "error", "score": 0.5}

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

@sentiment_bp.route('/diagnostic', methods=['GET'])
def diagnostic():
    result = {}

    # Check nltk data directories
    nltk_dirs = [
        '/app/nltk_data',
        '/opt/venv/nltk_data',
        '/root/nltk_data'
    ]

    for directory in nltk_dirs:
        if os.path.exists(directory):
            result[directory] = {
                'exists': True,
                'contents': os.listdir(directory)
            }

            # Check for punkt
            punkt_dir = os.path.join(directory, 'tokenizers', 'punkt')
            if os.path.exists(punkt_dir):
                result[directory]['punkt'] = {
                    'exists': True,
                    'contents': os.listdir(punkt_dir)
                }
            else:
                result[directory]['punkt'] = {'exists': False}
        else:
            result[directory] = {'exists': False}

    # Check nltk data paths
    import nltk
    result['nltk_data_path'] = nltk.data.path

    return jsonify(result)


@sentiment_bp.route('/setup-nltk', methods=['GET'])
def setup_nltk():
    try:
        # Run your setup code
        import nltk
        import ssl
        import os

        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Create a directory for NLTK data
        nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)

        # Set the NLTK data path
        nltk.data.path.append(nltk_data_dir)

        # Download NLTK resources
        nltk.download('punkt', download_dir=nltk_data_dir)
        nltk.download('stopwords', download_dir=nltk_data_dir)
        nltk.download('wordnet', download_dir=nltk_data_dir)

        # Check directory structure
        result = {
            "success": True,
            "nltk_data_dir": nltk_data_dir,
            "nltk_data_path": nltk.data.path,
            "directory_exists": os.path.exists(nltk_data_dir),
            "contents": {}
        }

        # Check main directory contents
        if os.path.exists(nltk_data_dir):
            result["contents"]["main_dir"] = os.listdir(nltk_data_dir)

            # Check for punkt
            punkt_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
            if os.path.exists(punkt_dir):
                result["contents"]["punkt_dir"] = {
                    "exists": True,
                    "path": punkt_dir,
                    "contents": os.listdir(punkt_dir)
                }
            else:
                result["contents"]["punkt_dir"] = {
                    "exists": False,
                    "path": punkt_dir
                }

            # Check for stopwords
            stopwords_dir = os.path.join(nltk_data_dir, 'corpora', 'stopwords')
            if os.path.exists(stopwords_dir):
                result["contents"]["stopwords_dir"] = {
                    "exists": True,
                    "path": stopwords_dir,
                    "contents": os.listdir(stopwords_dir)
                }
            else:
                result["contents"]["stopwords_dir"] = {
                    "exists": False,
                    "path": stopwords_dir
                }

            # Check for wordnet
            wordnet_dir = os.path.join(nltk_data_dir, 'corpora', 'wordnet')
            if os.path.exists(wordnet_dir):
                result["contents"]["wordnet_dir"] = {
                    "exists": True,
                    "path": wordnet_dir,
                    "contents": os.listdir(wordnet_dir)[:10]  # Show first 10 files to avoid too much output
                }
            else:
                result["contents"]["wordnet_dir"] = {
                    "exists": False,
                    "path": wordnet_dir
                }

        # Test if we can access punkt tokenizer
        try:
            punkt_test = nltk.data.find('tokenizers/punkt')
            result["punkt_accessible"] = {
                "success": True,
                "path": str(punkt_test)
            }
        except LookupError as e:
            result["punkt_accessible"] = {
                "success": False,
                "error": str(e)
            }

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@sentiment_bp.route('/check-files', methods=['GET'])
def check_files():
    """Endpoint to check NLTK file structure"""
    result = {
        "environment": {
            "NLTK_DATA": os.environ.get('NLTK_DATA', 'Not set')
        },
        "directories": {}
    }

    # List contents of potential NLTK directories
    dirs_to_check = [
        '/app/nltk_data',
        '/opt/venv/nltk_data',
        '/root/nltk_data',
        '/usr/local/share/nltk_data',
        os.getcwd() + '/nltk_data'
    ]

    for directory in dirs_to_check:
        if os.path.exists(directory):
            try:
                # Get directory structure
                result["directories"][directory] = {
                    "exists": True,
                    "contents": os.listdir(directory)
                }

                # Check tokenizers/punkt specifically
                punkt_dir = os.path.join(directory, 'tokenizers', 'punkt')
                if os.path.exists(punkt_dir):
                    result["directories"][directory]["punkt"] = {
                        "exists": True,
                        "contents": os.listdir(punkt_dir)
                    }
                else:
                    result["directories"][directory]["punkt"] = {
                        "exists": False
                    }

                # Check for other NLTK directories
                for subdir in ['tokenizers', 'corpora', 'taggers']:
                    path = os.path.join(directory, subdir)
                    if os.path.exists(path):
                        result["directories"][directory][subdir] = {
                            "exists": True,
                            "contents": os.listdir(path)
                        }
            except Exception as e:
                result["directories"][directory] = {
                    "exists": True,
                    "error": str(e)
                }
        else:
            result["directories"][directory] = {
                "exists": False
            }

    # Also check NLTK's search path
    import nltk
    result["nltk_search_path"] = nltk.data.path

    # Try to find punkt
    try:
        punkt_path = nltk.data.find('tokenizers/punkt')
        result["punkt_test"] = {
            "success": True,
            "path": str(punkt_path)
        }
    except LookupError as e:
        result["punkt_test"] = {
            "success": False,
            "error": str(e)
        }

    return jsonify(result)