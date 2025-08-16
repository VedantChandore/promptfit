import os
import time
from promptfit.token_budget import estimate_tokens
from promptfit.optimizer import optimize_prompt
from promptfit.utils import split_sentences
from promptfit.embedder import get_embeddings
from promptfit.relevance import compute_cosine_similarities

# Setup (replace with your key if paraphrasing/compression is enabled)
os.environ["COHERE_API_KEY"] = "Kwi33HNnmXRDCkO4j7FndNP3LATOoKX3yvoOdztK"

# Example retrieved context from a RAG pipeline
retrieved_docs = """
You are a customer-support assistant. A user reports that their device fails
intermittently under cold conditions, the battery drains within two hours, and
previous support tickets went unanswered. They've provided logs and screenshots.
Please summarize the issues, note their emotional tone, propose immediate
fixes, and suggest long-term retention strategies.
"""

query = "Summarize issues, emotional tone, action items, and retention strategies."

# -------------------------------
# Benchmark: BEFORE PromptFit
# -------------------------------
print("=== BASELINE PROMPT ===")
print(retrieved_docs)

baseline_tokens = estimate_tokens(retrieved_docs)
print(f"\nBaseline Tokens: {baseline_tokens}\n")

# -------------------------------
# Benchmark: Relevance Scoring
# -------------------------------
sentences = split_sentences(retrieved_docs)
all_texts = [query] + sentences
embeddings = get_embeddings(all_texts)

query_emb = embeddings[0]
sent_embs = embeddings[1:]
scores = compute_cosine_similarities(query_emb, sent_embs)

print("=== RELEVANCE SCORES OF COSINE SIMILARITY ===")
for sent, score in zip(sentences, scores):
    print(f"{score:.4f} - {sent.strip()[:80]}...")

# -------------------------------
# Benchmark: AFTER PromptFit
# -------------------------------
budget = 40
start_time = time.time()
optimized_prompt = optimize_prompt(
    retrieved_docs, query, max_tokens=budget
)
optimization_time = time.time() - start_time

optimized_tokens = estimate_tokens(optimized_prompt)
tokens_saved = baseline_tokens - optimized_tokens
reduction_pct = tokens_saved / baseline_tokens * 100

print("\n=== EFFICIENCY STATS ===")
print(f"Original tokens: {baseline_tokens}")
print(f"Optimized tokens: {optimized_tokens}")
print(f"Tokens saved: {tokens_saved} ({reduction_pct:.1f}% reduction)")
print(f"Optimization time: {optimization_time:.2f} seconds")

print("\n=== OPTIMIZED PROMPT ===")
print(optimized_prompt)
