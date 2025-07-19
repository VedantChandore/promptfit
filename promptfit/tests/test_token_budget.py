# test_token_budget.py
# Unit tests for token_budget module

import pytest
from promptfit import token_budget

# Test fallback token estimation (no Cohere)
def test_estimate_tokens_fallback():
    text = "one two three four five six seven eight nine ten"  # 10 words
    tokens = token_budget.estimate_tokens(text)
    # 10 words / 0.75 ≈ 13 tokens
    assert tokens == 13

def test_estimate_tokens_per_section():
    sections = ["First section.", "Second section is longer."]
    tokens = token_budget.estimate_tokens_per_section(sections)
    assert isinstance(tokens, list)
    assert len(tokens) == 2
    assert all(isinstance(t, int) for t in tokens)

def test_estimate_total_tokens():
    sections = ["One.", "Two three.", "Four five six."]
    total = token_budget.estimate_total_tokens(sections)
    # Each section: 1/0.75=1, 2/0.75≈2, 3/0.75=4; sum=7
    assert total == 7

# Optionally, add a test for Cohere if API key is present and cohere is installed
@pytest.mark.skipif(not token_budget._has_cohere, reason="Cohere not installed")
def test_estimate_tokens_with_cohere(monkeypatch):
    class DummyResp:
        tokens = [1,2,3]
    class DummyCohere:
        def __init__(self, key): pass
        def tokenize(self, text): return DummyResp()
    monkeypatch.setattr(token_budget, "cohere", type("cohere", (), {"Client": DummyCohere}))
    monkeypatch.setattr(token_budget, "_has_cohere", True)
    monkeypatch.setattr("promptfit.utils.get_cohere_api_key", lambda: "dummy")
    tokens = token_budget.estimate_tokens("foo bar baz")
    assert tokens == 3 