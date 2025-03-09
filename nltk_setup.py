import nltk
import os

# Set the NLTK data path to the local directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Set the path for NLTK to look for data
nltk.data.path.append(nltk_data_dir)

# Download the required data
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)