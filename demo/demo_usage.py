# demo_relevance_on_prompt.py

import os

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

# 3. Split into sentences
sentences = split_sentences(prompt)
print("=== SENTENCES ===")
for i, s in enumerate(sentences, 1):
    print(f"{i}: {s!r}")
print()

# 4. Compute embeddings (first the query, then the sentences)
all_texts = [query] + sentences
embs = get_embeddings(all_texts)

# 5. Compute cosine similarities between query and each sentence
query_emb = embs[0]
sent_embs = embs[1:]
scores = compute_cosine_similarities(query_emb, sent_embs)

# 6. Display relevance scores
print("=== RELEVANCE SCORES ===")
for sent, score in zip(sentences, scores):
    print(f"{score:.4f} – {sent!r}")
print()

# 7. Show total token count before optimization
orig_tokens = estimate_tokens(prompt)
print(f"Original prompt ≈ {orig_tokens} tokens\n")

# 8. Run optimizer
budget = 40
optimized = optimize_prompt(prompt, query, max_tokens=budget)
opt_tokens = estimate_tokens(optimized)

# 9. Output final result
print(f"Optimized prompt ({opt_tokens} tokens ≤ {budget} budget):\n")
print(optimized)
