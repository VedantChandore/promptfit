import os
from dotenv import load_dotenv
import nltk

from .config import COHERE_API_KEY_ENV

# Always load .env from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, '.env'))


def get_cohere_api_key():
    """Load Cohere API key from environment or .env file."""
    key = os.getenv(COHERE_API_KEY_ENV)
    if not key:
        raise ValueError("Cohere API key not found. Set COHERE_API_KEY in your environment or .env file.")
    return key

# Sentence splitting utility
def split_sentences(text):
    """Split text into sentences using nltk. Ensures punkt and punkt_tab are available."""
    for resource in ['punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)
    return nltk.sent_tokenize(text) 