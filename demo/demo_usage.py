# demo_relevance_on_prompt.py

import os
import time

# 1. Set your Cohere key (or have it in your env)
os.environ["COHERE_API_KEY"] = "Kwi33HNnmXRDCkO4j7FndNP3LATOoKX3yvoOdztK"

from promptfit.token_budget import estimate_tokens
from promptfit.utils import split_sentences
from promptfit.embedder import get_embeddings
from promptfit.relevance import compute_cosine_similarities
from promptfit.optimizer import optimize_prompt

# 2. Your actual long prompt
prompt = """
You are a customer-support assistant. A user reports that their device fails
intermittently under cold conditions, the battery drains within two hours, and
previous support tickets went unanswered. They’ve provided logs and screenshots.
Please summarize the issues, note their emotional tone, propose immediate
fixes, and suggest long-term retention strategies.
"""

query = "Summarize issues, emotional tone, action items, and retention strategies."

# Pre-compute token counts and optimization
orig_tokens = estimate_tokens(prompt)
budget = 40
start_time = time.time()
optimized = optimize_prompt(prompt, query, max_tokens=budget)
end_time = time.time()
opt_tokens = estimate_tokens(optimized)

tokens_saved = orig_tokens - opt_tokens
percent_saved = (tokens_saved / orig_tokens) * 100
elapsed_time = end_time - start_time

# 3. Efficiency Stats FIRST
print("--- Efficiency Stats ---")_2
print(f"Original tokens: {orig_tokens}")
print(f"Optimized tokens: {opt_tokens}")
print(f"Tokens saved: {tokens_saved} ({percent_saved:.1f}% reduction)")
print(f"Optimization time: {elapsed_time:.2f} seconds")
print()

# 4. Display prompt and analysis
print("=== PROMPT ===")
print(prompt)

# 5. Split into sentences
sentences = split_sentences(prompt)
print("=== SENTENCES ===")
for i, s in enumerate(sentences, 1):
    print(f"{i}: {s!r}")
print()

# 6. Compute embeddings and relevance
all_texts = [query] + sentences
embs = get_embeddings(all_texts)

query_emb = embs[0]
sent_embs = embs[1:]
scores = compute_cosine_similarities(query_emb, sent_embs)

print("=== RELEVANCE SCORES OF COSINE SIMILARITY===")
for sent, score in zip(sentences, scores):
    print(f"{score:.4f}{sent!r}")
print()

# 7. Print optimized prompt
print(f"Optimized prompt ({opt_tokens} tokens <= {budget} budget):\n")
print(optimized)
