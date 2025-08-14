import pytest # type: ignore
from promptfit import paraphraser

@pytest.mark.skipif(paraphraser.cohere is None, reason="cohere not installed")
def test_paraphrase_prompt(monkeypatch):
    class DummyGen:
        def __init__(self, text):
            self.text = text

    class DummyResp:
        def __init__(self, text):
            self.generations = [DummyGen(text)]

    class DummyCohere:
        def __init__(self, key):
            pass
        def generate(self, **kwargs):
            return DummyResp("compressed prompt")

    monkeypatch.setattr(paraphraser, "cohere", type("cohere", (), {"Client": DummyCohere}))
    monkeypatch.setattr(paraphraser, "get_cohere_api_key", lambda: "dummy")

    result = paraphraser.paraphrase_prompt("long prompt", instructions="shorten", max_tokens=10)
    assert result == "compressed prompt"
