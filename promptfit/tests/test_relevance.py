# test_relevance.py
# Unit tests for relevance module

import numpy as np
from promptfit import relevance

def test_compute_cosine_similarities_basic():
    ref = [1.0, 0.0]
    segs = [[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]]
    sims = relevance.compute_cosine_similarities(ref, segs)
    # First is 1.0 (identical), second is 0.0 (orthogonal), third is ~0.707
    assert np.isclose(sims[0], 1.0)
    assert np.isclose(sims[1], 0.0)
    assert np.isclose(sims[2], 0.707, atol=0.01)

def test_rank_segments_by_relevance():
    segments = ["A", "B", "C"]
    reference = "REF"
    # Fake embedding function: returns index as embedding
    def fake_get_embeddings(texts):
        # ref: [1,0,0], A: [0,1,0], B: [0,0,1], C: [1,1,1]
        return [[1,0,0], [0,1,0], [0,0,1], [1,1,1]]
    ranked = relevance.rank_segments_by_relevance(segments, reference, fake_get_embeddings)
    # Should return a list of (segment, score) tuples, sorted by score desc
    assert isinstance(ranked, list)
    assert all(isinstance(x, tuple) and isinstance(x[0], str) and isinstance(x[1], float) for x in ranked)
    # Highest similarity should be C (dot with ref is 1), then A/B (dot is 0)
    assert ranked[0][0] == "C" 