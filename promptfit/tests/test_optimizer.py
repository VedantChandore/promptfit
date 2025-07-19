# test_optimizer.py
# Unit tests for optimizer module

from promptfit import optimizer

def test_optimize_prompt_basic(monkeypatch):
    # Mock all dependencies
    monkeypatch.setattr(optimizer, "split_sentences", lambda text: ["A", "B", "C"])
    monkeypatch.setattr(optimizer, "estimate_tokens_per_section", lambda sections: [10, 20, 30])
    monkeypatch.setattr(optimizer, "estimate_tokens", lambda s: 10 if s == "A" else 20 if s == "B" else 30)
    monkeypatch.setattr(optimizer, "rank_segments_by_relevance", lambda sections, query, get_emb: [(s, 1.0) for s in sections[::-1]])
    monkeypatch.setattr(optimizer, "paraphrase_prompt", lambda prompt, instructions, max_tokens: "PARAPHRASED")
    # Case 1: Under budget
    result = optimizer.optimize_prompt("irrelevant", "query", max_tokens=100)
    # All sections: 10+20+30=60 < 100, so should return original prompt
    assert result == "irrelevant"
    # Case 2: Over budget, triggers pruning and paraphrasing
    def fake_estimate_tokens(s):
        return 50 if s == "A" else 60
    monkeypatch.setattr(optimizer, "estimate_tokens", fake_estimate_tokens)
    monkeypatch.setattr(optimizer, "estimate_tokens_per_section", lambda sections: [50, 60, 60])
    result = optimizer.optimize_prompt("irrelevant", "query", max_tokens=50)
    # Only "A" fits, but estimate_tokens("A") == 50, so paraphrasing is triggered
    assert result == "PARAPHRASED" 