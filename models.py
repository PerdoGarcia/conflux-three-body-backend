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

def load_local_sentiment_model():
    """Load sentiment model from local directory"""
    logger.info("Loading sentiment model from local directory...")
    try:
        # Update these paths to match your local model directories
        model_dir = 'local_models/sentiment_model'
        tokenizer_dir = 'local_models/sentiment_tokenizer'

        # Load models
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        sentiment_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        logger.info("Sentiment model loaded successfully from local directory")
        return sentiment_model, sentiment_tokenizer
    except Exception as e:
        logger.error(f"Error loading sentiment model from local directory: {e}")
        raise

def load_local_emotion_model():
    """Load emotion model from local directory"""
    logger.info("Loading emotion model from local directory...")
    try:
        # Update these paths to match your local model directories
        model_dir = 'local_models/emotion_model'
        tokenizer_dir = 'local_models/emotion_tokenizer'

        # Load models
        emotion_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        emotion_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)

        logger.info("Emotion model loaded successfully from local directory")
        return emotion_model, emotion_tokenizer
    except Exception as e:
        logger.error(f"Error loading emotion model from local directory: {e}")
        raise

def load_local_sbert_model():
    """Load SBERT model from local directory"""
    logger.info("Loading SBERT model from local directory...")
    try:
        # Update this path to match your local model directory
        model_dir = 'local_models/sbert'

        # Load model
        sbert_model = SentenceTransformer(model_dir)

        logger.info("SBERT model loaded successfully from local directory")
        return sbert_model
    except Exception as e:
        logger.error(f"Error loading SBERT model from local directory: {e}")
        raise

def initialize_models():
    """Initialize all models from local directories"""
    global sentiment_pipeline, emotion_pipeline, sbert_model, rake_extractor

    logger.info("Loading NLP models from local directories...")

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

        # Load models from local directories
        try:
            # Load sentiment model
            sentiment_model, sentiment_tokenizer = load_local_sentiment_model()
            print("done with sentiment model")

            # Load emotion model
            emotion_model, emotion_tokenizer = load_local_emotion_model()
            print("done with emotional model")

            # Load SBERT model
            sbert_model = load_local_sbert_model()
            print("done with sbert model")

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

            logger.info("All models loaded successfully from local directories!")
            return True

        except Exception as e:
            logger.error(f"Error loading models from local directories: {e}")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")