# embedder.py
# Generate and cache embeddings using Cohere 
from typing import List, Dict

try:
    import cohere
except ImportError:
    cohere = None

from .utils import get_cohere_api_key
from .config import COHERE_EMBED_MODEL

# Simple in-memory cache for embeddings
_embedding_cache: Dict[str, List[float]] = {}

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts using Cohere. Uses in-memory cache."""
    if cohere is None:
        raise ImportError("cohere package is required for embedding generation.")
    api_key = get_cohere_api_key()
    co = cohere.Client(api_key)
    uncached = [t for t in texts if t not in _embedding_cache]
    if uncached:
        response = co.embed(
            texts=uncached,
            model=COHERE_EMBED_MODEL,
            input_type="search_document"  # Required for embed-english-v3.0
        )
        for text, emb in zip(uncached, response.embeddings):
            _embedding_cache[text] = emb
    return [_embedding_cache[t] for t in texts] 