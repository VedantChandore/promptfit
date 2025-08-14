# demo_rag.py
import faiss # type: ignore
import numpy as np # type: ignore

from promptfit.embedder import get_embeddings
from promptfit.relevance import rank_segments_by_relevance
from promptfit.optimizer import optimize_prompt
from promptfit.token_budget import estimate_total_tokens

# ===== Step 1: Load documents =====
documents = [
    "Python is a high-level programming language that supports multiple paradigms.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Cohere provides APIs for natural language processing tasks like embeddings and generation.",
    "Retrieval-Augmented Generation (RAG) combines search and LLMs to answer queries with external knowledge.",
]

# ===== Step 2: Build FAISS index =====
embs = np.array(get_embeddings(documents)).astype("float32")
index = faiss.IndexFlatIP(embs.shape[1])  # Inner product for cosine sim after normalization
faiss.normalize_L2(embs)  # Cosine similarity
index.add(embs)

# ===== Step 3: Query =====
query = "How can I use FAISS for retrieval in RAG?"
query_emb = np.array(get_embeddings([query])).astype("float32")
faiss.normalize_L2(query_emb)
D, I = index.search(query_emb, k=3)

retrieved_docs = [documents[i] for i in I[0]]
print("\n[Retrieved Docs]")
for doc in retrieved_docs:
    print("-", doc)

# ===== Step 4: Rank retrieved docs by relevance =====
ranked_docs = rank_segments_by_relevance(
    retrieved_docs,
    query,
    get_embeddings,
    top_k=3
)

print("\n[Ranked Docs]")
for doc, score in ranked_docs:
    print(f"{score:.4f} | {doc}")

# ===== Step 5: Optimize prompt for LLM =====
context = "\n".join(doc for doc, _ in ranked_docs)
prompt = f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {query}"
optimized_prompt = optimize_prompt(prompt, query, max_tokens=150)

print("\n[Optimized Prompt]")
print(optimized_prompt)
print(f"Token Count: {estimate_total_tokens([optimized_prompt])}")

# ===== Step 6: Mock LLM call (replace with real Cohere/OpenAI call) =====
print("\n[Mock Answer]")
print("FAISS can be used in RAG by storing document embeddings and retrieving the most relevant ones for a query.")
