# backend/fixed_senti.py
import json
import sys
import nltk
import time
import numpy as np
from flask import Blueprint, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rake_nltk import Rake
import os

# Create the Blueprint
sentiment_bp = Blueprint('sentiment', __name__)

# Global variables to store models and tokenizers
sentiment_pipeline = None
emotion_pipeline = None
sbert_model = None
rake_extractor = None
stored_texts = []

def initialize_models():
    """Initialize all models from disk or download if needed"""
    global sentiment_pipeline, emotion_pipeline, sbert_model, rake_extractor

    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)

    print("Loading models...")

    try:
        # Check if local model directories exist and have content
        sentiment_model_exists = os.path.exists(f"{model_dir}/sentiment_model") and any(os.listdir(f"{model_dir}/sentiment_model"))
        sentiment_tokenizer_exists = os.path.exists(f"{model_dir}/sentiment_tokenizer") and any(os.listdir(f"{model_dir}/sentiment_tokenizer"))
        emotion_model_exists = os.path.exists(f"{model_dir}/emotion_model") and any(os.listdir(f"{model_dir}/emotion_model"))
        emotion_tokenizer_exists = os.path.exists(f"{model_dir}/emotion_tokenizer") and any(os.listdir(f"{model_dir}/emotion_tokenizer"))
        sbert_exists = os.path.exists(f"{model_dir}/sbert") and any(os.listdir(f"{model_dir}/sbert"))

        # Load sentiment model (local or download)
        if sentiment_model_exists and sentiment_tokenizer_exists:
            print("Loading sentiment model from local files...")
            sentiment_model = AutoModelForSequenceClassification.from_pretrained(f"{model_dir}/sentiment_model", local_files_only=True)
            sentiment_tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/sentiment_tokenizer", local_files_only=True)
        else:
            print("Downloading sentiment model from Hugging Face...")
            sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
            sentiment_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            # Save for future use
            sentiment_model.save_pretrained(f"{model_dir}/sentiment_model")
            sentiment_tokenizer.save_pretrained(f"{model_dir}/sentiment_tokenizer")

        # Load emotion model (local or download)
        if emotion_model_exists and emotion_tokenizer_exists:
            print("Loading emotion model from local files...")
            emotion_model = AutoModelForSequenceClassification.from_pretrained(f"{model_dir}/emotion_model", local_files_only=True)
            emotion_tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/emotion_tokenizer", local_files_only=True)
        else:
            print("Downloading emotion model from Hugging Face...")
            emotion_model = AutoModelForSequenceClassification.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")
            emotion_tokenizer = AutoTokenizer.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student", use_fast=False)
            # Save for future use
            emotion_model.save_pretrained(f"{model_dir}/emotion_model")
            emotion_tokenizer.save_pretrained(f"{model_dir}/emotion_tokenizer")

        # Load sentence transformer (local or download)
        if sbert_exists:
            print("Loading sentence transformer from local files...")
            sbert_model = SentenceTransformer(f"{model_dir}/sbert")
        else:
            print("Downloading sentence transformer from Hugging Face...")
            sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            # Save for future use
            sbert_model.save(f"{model_dir}/sbert")

        # Initialize pipelines
        device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        sentiment_pipeline = pipeline("text-classification", model=sentiment_model, tokenizer=sentiment_tokenizer, device=device)
        emotion_pipeline = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer, top_k=3, device=device)

        # Initialize RAKE
        rake_extractor = Rake()

        print("All models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        # If there was an error with local files, try direct download without saving
        try:
            print("Attempting to load models directly from Hugging Face...")
            sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
            sentiment_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            sentiment_pipeline = pipeline("text-classification", model=sentiment_model, tokenizer=sentiment_tokenizer)

            emotion_model = AutoModelForSequenceClassification.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")
            emotion_tokenizer = AutoTokenizer.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student", use_fast=False)
            emotion_pipeline = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer, top_k=3)

            sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

            rake_extractor = Rake()

            print("Models loaded directly from Hugging Face!")
            return True
        except Exception as inner_e:
            print(f"Failed to load models directly from Hugging Face: {inner_e}")
            raise

def download_nltk_resources():
    """Download required NLTK resources"""
    try:
        # Download stopwords if not already present
        nltk.download('stopwords')
        # Download punkt tokenizer if not already present
        nltk.download('punkt')
        print("NLTK resources downloaded successfully")
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")

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
    try:
        # Force JSON parsing in case Content-Type is not set properly
        data = request.get_json(force=True)
        message_text = data.get('message_text', '')

        # stored_texts is expected to be a list of objects, e.g., [{ "id": 1, "text": "..." }]
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

        # Build the final response object (merging the analysis results with the new message data)
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

        print("Processed result:", response_obj)

        # Use NumpyEncoder to handle numpy types
        return json.dumps(response_obj, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        print(f"Error in get_sentiment: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def get_sentiment_analysis(text):
    """Analyze sentiment using the pre-loaded pipeline"""
    global sentiment_pipeline
    result = sentiment_pipeline(text)[0]
    return {"label": result['label'], "score": float(result['score'])}

def get_emotion_analysis(text):
    """Analyze emotions using the pre-loaded pipeline"""
    global emotion_pipeline
    emotions = emotion_pipeline(text)[0]
    return {e['label']: float(e['score']) for e in emotions}

def get_intensity(emotion_scores):
    """Calculate emotional intensity from emotion scores"""
    if not emotion_scores:
        return 0.0
    return float(max(emotion_scores.values()))

def extract_keywords(text, num_keywords=5):
    """Extract keywords using pre-loaded RAKE"""
    global rake_extractor
    rake_extractor.extract_keywords_from_text(text)
    phrases = rake_extractor.get_ranked_phrases()
    return phrases[:num_keywords] if phrases else []

def get_text_similarity(text1, text2):
    """Calculate text similarity using sentence embeddings"""
    global sbert_model
    try:
        embeddings = sbert_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Error calculating text similarity: {e}")
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
        print(f"Error calculating closeness: {e}")
        return {
            "closeness_score": 0.0,
            "emotion_similarity": 0.0,
            "text_similarity": 0.0,
            "intensity_similarity": 0.0
        }

def process_input(text, stored_inputs):
    """Process a single text input and return analysis results"""
    print(f"Processing input: {text}")

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
                print(f"Error calculating closeness: {e}")
                continue

        if highest_closeness_data:
            result["highest_closeness_id"] = highest_closeness_id
            result["closeness_score"] = highest_closeness_data["closeness_score"]
            result["emotion_similarity"] = highest_closeness_data["emotion_similarity"]
            result["text_similarity"] = highest_closeness_data["text_similarity"]
            result["intensity_similarity"] = highest_closeness_data["intensity_similarity"]

    return result

# Initialize models
# download_nltk_resources()
initialize_models()