from typing import List, Tuple, Callable, Optional, Sequence
import numpy as np # type: ignore
from sklearn.metrics.pairwise import cosine_similarity

def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms

def _call_get_embeddings_batched(texts: Sequence[str], get_embeddings_fn: Callable[[List[str]], List[List[float]]], batch_size: int) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embs = get_embeddings_fn(list(batch))
        if batch_embs is None:
            raise ValueError("get_embeddings_fn returned None for batch")
        all_embs.extend(batch_embs)
    arr = np.array(all_embs)
    if arr.ndim != 2:
        raise ValueError("Embeddings must be 2D (n_texts, dim)")
    return arr

def compute_cosine_similarities(reference_emb: Sequence[float], segment_embs: Sequence[Sequence[float]]) -> List[float]:
    ref = np.array(reference_emb).reshape(1, -1)
    segs = np.array(segment_embs)
    if ref.shape[1] != segs.shape[1]:
        raise ValueError(f"Dimension mismatch: reference dim {ref.shape[1]} vs segment dim {segs.shape[1]}")
    ref_n = _l2_normalize(ref)
    segs_n = _l2_normalize(segs)
    sims = float(ref_n.dot(segs_n.T)) if segs_n.shape[0] == 1 else (ref_n @ segs_n.T)[0]
    return sims.tolist() if isinstance(sims, np.ndarray) else [sims] # type: ignore

def rank_segments_by_relevance(
    segments: List[str],
    reference: str,
    get_embeddings_fn: Callable[[List[str]], List[List[float]]],
    *,
    top_k: Optional[int] = None,
    batch_size: int = 128,
    hyde: bool = False,
    llm_expand_fn: Optional[Callable[[str], str]] = None
) -> List[Tuple[str, float]]:
    if not isinstance(segments, list):
        raise TypeError("segments must be a list of strings")
    if reference is None:
        raise ValueError("reference must not be None")
    if len(segments) == 0:
        return []

    if hyde and llm_expand_fn is None:
        raise ValueError("hyde=True requires an llm_expand_fn")

    reference_for_embedding = reference
    if hyde:
        reference_for_embedding = llm_expand_fn(reference_for_embedding) # type: ignore

    texts = [reference_for_embedding] + segments
    embs = _call_get_embeddings_batched(texts, get_embeddings_fn, batch_size)
    ref_emb = embs[0]
    seg_embs = embs[1:]

    if seg_embs.shape[0] != len(segments):
        raise ValueError("Number of segment embeddings does not match number of segments")

    # normalize and compute similarities efficiently
    ref_n = _l2_normalize(ref_emb.reshape(1, -1))
    segs_n = _l2_normalize(seg_embs)
    sims = (ref_n @ segs_n.T)[0]

    pairs = list(zip(segments, sims.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        return pairs[:top_k]
    return pairs
