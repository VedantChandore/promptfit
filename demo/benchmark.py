# verify_promptfit.py

import os
import sys
from promptfit.token_budget import estimate_tokens, estimate_tokens_per_section, estimate_total_tokens
from promptfit.relevance import rank_segments_by_relevance
from promptfit.optimizer import optimize_prompt

# Make sure your COHERE_API_KEY is set
if not os.getenv("COHERE_API_KEY"):
    print("⚠️  WARNING: COHERE_API_KEY not found in environment. Some tests will be skipped.")

def test_token_budget():
    text = "This is a simple test sentence."
    tokens = estimate_tokens(text)
    print(f"Token count for \"{text}\": {tokens}")

def test_relevance():
    segments = ["apple", "banana", "cherry"]
    ref = "I like bananas and fruit."
    ranked = rank_segments_by_relevance(segments, ref, get_embeddings_fn=lambda texts: [[0]] * len(texts))
    print("Relevance ranking (mocked embeddings):", ranked)

def test_optimizer():
    prompt = "A B C D E F G H I J"
    query = "pick letters"
    # set budget very low so pruning+paraphrase triggers
    optimized = optimize_prompt(prompt, query, max_tokens=5)
    tok = estimate_tokens(optimized)
    print(f"Optimized prompt: \"{optimized}\" ({tok} tokens)")
    assert tok <= 5, f"Budget exceeded: got {tok} tokens"

if __name__ == "__main__":
    print("=== TOKEN BUDGET TEST ===")
    test_token_budget()
    print("\n=== RELEVANCE TEST ===")
    test_relevance()
    print("\n=== OPTIMIZER TEST ===")
    try:
        test_optimizer()
        print("Optimizer test passed ✅")
    except AssertionError as e:
        print("Optimizer test failed ❌", e)
        sys.exit(1)
    print("\nAll manual checks completed.")
