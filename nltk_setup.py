import nltk
import os

# Create NLTK data directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Set the path for NLTK to look for data
nltk.data.path.append(nltk_data_dir)

# Download only English stopwords
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)

# Keep only English stopwords to reduce size
stopwords_dir = os.path.join(nltk_data_dir, 'corpora', 'stopwords')
if os.path.exists(stopwords_dir):
    # Get the list of all language files
    all_languages = [f for f in os.listdir(stopwords_dir) if not f.startswith('.')]

    # Keep only English
    for lang_file in all_languages:
        if lang_file != 'english':
            try:
                lang_path = os.path.join(stopwords_dir, lang_file)
                if os.path.isfile(lang_path):
                    os.remove(lang_path)
                    print(f"Removed stopwords for: {lang_file}")
            except Exception as e:
                print(f"Could not remove {lang_file}: {e}")

print("NLTK setup complete with English resources only")