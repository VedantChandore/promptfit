# promptfit

[![PyPI version](https://badge.fury.io/py/promptfit.svg)](https://pypi.org/project/promptfit/)
[![GitHub stars](https://img.shields.io/github/stars/VedantChandore/promptfit?style=social)](https://github.com/VedantChandore/promptfit)


---

---

## 📣 Author

**Vedant Laxman Chandore**  
[GitHub](https://github.com/VedantChandore)

##  The Core Problem
Modern LLMs (Cohere, OpenAI, Gemini, Anthropic, etc.) are powerful, but their **token limits** make it hard to fit rich, multi-section prompts—especially for Retrieval-Augmented Generation (RAG), few-shot learning, and instruction-heavy use cases. Developers waste time manually trimming prompts, risking loss of important context, incomplete responses, or costly token overages.

**promptfit** solves this by automating prompt analysis, compression, and optimization—so you get the most value from every token, every time.

---

## ✨ Features

- **🔢 Token Budget Estimator:** Analyze and estimate token usage for prompt templates, sections, and variables—before sending to an LLM.
- **🧭 Semantic Relevance Scoring:** Split prompts into sections, generate embeddings (Cohere), and rank by cosine similarity to your query or task.
- **✂️ Smart Prompt Pruner:** Drop or trim low-salience sections first, keeping only the most relevant content to fit your token budget.
- **✍️ Paraphrasing Module:** Use Cohere’s LLM to rewrite and compress over-budget prompts, preserving key instructions and meaning.
- **📦 Modular Design:** Each feature is a standalone module—use them independently or together.
- **🧪 Test-Driven:** Comprehensive unit tests with mocked or live Cohere API responses.
- **🔐 Secure API Key Handling:** Loads your Cohere API key from a `.env` file or environment variable.
- **🖥️ CLI Support:** Optimize prompts directly from the command line.
---
✅ Does promptfit Lose Context?
❌ If Done Naively:
If you just truncate text to meet token limits, yes, you risk removing vital context.

This is what most LLM developers currently do manually — and it's dangerous.

✅ What promptfit Does Instead:
Your package intelligently retains semantically relevant chunks and:

✅ Uses cosine similarity between the query and each sentence to prioritize important information.

✅ Applies token estimation to make sure output fits within budget.

✅ Optionally uses paraphrasing (via Cohere LLM) to compress rather than drop content.

✅ Keeps the query in mind throughout — relevance is measured with respect to the query, not blindly.


---
## 📊 PromptFit vs Baseline Comparison (RAG Flow)
|                         Metric | Without PromptFit (Baseline) | With PromptFit (Optimized) | Improvement     |
| -----------------------------: | :--------------------------: | :------------------------: | :-------------- |
| 🔢 Tokens in Retrieved Context |              284             |             97             | ↓ 65.8%         |
|           ⏱️ Optimization Time |              N/A             |            1.72s           | \~Real-time     |
|              💬 Prompt Clarity |  Mixed, sometimes redundant  |     Concise & relevant     | ✅ More focused  |
|             💰 Cost Efficiency | Higher (due to long context) |    Lower (fewer tokens)    | ↓ Reduced spend |
|        📈 LLM Response Quality |       Slightly verbose       |     Direct, contextual     | ✅ More precise  |

✅ PromptFit ensures you never go over token budget, preserves semantic relevance, and boosts LLM efficiency for production-ready GenAI apps.

---
## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **LLM:** [Cohere command-r-plus](https://docs.cohere.com/docs/models-overview)
- **Embeddings:** embed-english-v3.0
- **Tokenizer:** Cohere’s estimator (or manual fallback)
- **Libraries:**
  - `cohere`, `scikit-learn`, `tiktoken`, `python-dotenv`, `nltk`/`spacy`, `rich`, `typer`, `pytest`

---

## 📦 Installation

```bash
pip install promptfit
```

---
## Tested Output

![image_alt](https://github.com/VedantChandore/promptfit/blob/main/docs/image.png?raw=true)


## 🚀 Demo Usage

### **Python API**

```python
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

print("=== PROMPT ===")
print(prompt)

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
print("=== RELEVANCE SCORES OF COSINE SIMILARITY===")
for sent, score in zip(sentences, scores):
    print(f"{score:.4f} – {sent!r}")
print()

# 7. Show total token count before optimization
orig_tokens = estimate_tokens(prompt)
print(f"Original prompt ≈ {orig_tokens} tokens\n")

# 8. Run optimizer with timing
budget = 40
start_time = time.time()
optimized = optimize_prompt(prompt, query, max_tokens=budget)
end_time = time.time()
opt_tokens = estimate_tokens(optimized)

tokens_saved = orig_tokens - opt_tokens
percent_saved = (tokens_saved / orig_tokens) * 100

# 9. Output final result with efficiency stats
print(f"Optimized prompt ({opt_tokens} tokens ≤ {budget} budget):\n")
print(optimized)
print("\n--- Efficiency Stats ---")
print(f"Token reduction: {orig_tokens - opt_tokens} tokens")
print(f"Reduction percentage: {(orig_tokens - opt_tokens) / orig_tokens * 100:.1f}%")
print(f"Optimization time: {end_time - start_time:.2f} seconds")
print(f"Tokens Saved: {tokens_saved}")

```

### **Command Line**

```bash
python -m promptfit.cli "YOUR_PROMPT" "YOUR_QUERY" --max-tokens 120
```

### **Full Demo Script**
See [`demo/demo_usage.py`](demo/demo_usage.py) for a comprehensive example covering:
- Token estimation
- Embedding generation
- Relevance ranking
- Pruning and paraphrasing
- End-to-end optimization

---

## 🗝️ Environment Setup

- Store your `COHERE_API_KEY` in a `.env` file in your project root:
  ```
  COHERE_API_KEY=your-real-api-key-here
  ```
- Or set it in your shell:
  ```bash
  export COHERE_API_KEY=your-real-api-key-here
  ```

---

## 📚 Directory Structure

```
promptfit/
│
├── __init__.py
├── token_budget.py
├── embedder.py
├── relevance.py
├── optimizer.py
├── paraphraser.py
├── cli.py
├── utils.py
├── config.py
│
├── README.md
├── requirements.txt
│
├── tests/
│   ├── test_token_budget.py
│   ├── test_relevance.py
│   ├── test_optimizer.py
│   └── test_paraphraser.py
│
└── demo/
    └── demo_usage.py
```

---

## 💡 Why Use promptfit?

- **Save tokens, save money:** Only send the most relevant, concise prompts to your LLM.
- **Prevent errors:** Never exceed token limits or lose critical context.
- **Automate prompt engineering:** Focus on your app, not manual prompt trimming.
- **Works with any LLM:** Designed for Cohere, but easily adaptable to OpenAI, Gemini, Anthropic, and more.

---

## 📝 License

MIT License

---

## 🤝 Contributing

Pull requests, issues, and stars are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

## 📣 Author

**Vedant Laxman Chandore**  
[GitHub](https://github.com/VedantChandore)

---


*Built for the next generation of GenAI and LLM developers. Optimize your prompts, maximize your results!* 
=======


