# relevance.py
# Cosine similarity between query and prompt segments 
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def compute_cosine_similarities(reference_emb: List[float], segment_embs: List[List[float]]) -> List[float]:
    """Compute cosine similarity between reference embedding and each segment embedding."""
    ref = np.array(reference_emb).reshape(1, -1)
    segs = np.array(segment_embs)
    sims = cosine_similarity(ref, segs)[0]
    return sims.tolist()


def rank_segments_by_relevance(segments: List[str], reference: str, get_embeddings_fn) -> List[Tuple[str, float]]:
    """Rank segments by cosine similarity to the reference string."""
    all_texts = [reference] + segments
    embs = get_embeddings_fn(all_texts)
    ref_emb = embs[0]
    seg_embs = embs[1:]
    sims = compute_cosine_similarities(ref_emb, seg_embs)
    ranked = sorted(zip(segments, sims), key=lambda x: x[1], reverse=True)
    return ranked 