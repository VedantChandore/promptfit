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

## 🚀 Demo Usage

### **Python API**

```python
from promptfit import optimize_prompt

=======
from promptfit.token_budget import estimate_tokens, estimate_tokens_per_section, estimate_total_tokens
from promptfit.embedder import get_embeddings
from promptfit.relevance import rank_segments_by_relevance
from promptfit.paraphraser import paraphrase_prompt
from promptfit.utils import split_sentences


query = "Summarize this support ticket with action items."
prompt = """
You are a helpful assistant. This is the customer's full complaint about a recent product issue. The customer is frustrated and has provided a lot of background information, some of which may not be relevant. Please include background, emotional tone, and recommendations in your summary. Make sure to be concise and actionable.

Background: The customer purchased the product two months ago and has experienced intermittent issues since then. They have contacted support twice before but did not receive a satisfactory resolution. The customer is now requesting a refund or a replacement.

Details: The product fails to start on cold mornings, and the battery drains quickly. The customer has tried all troubleshooting steps provided in the manual. They are upset about the lack of response from support and mention that they may leave a negative review if the issue is not resolved soon.
"""

<<<<<<< HEAD
optimized = optimize_prompt(prompt, query, max_tokens=120)
print("Optimized prompt:\n", optimized)
=======
print("--- TOKEN ESTIMATION ---")
sections = split_sentences(prompt)
tokens_per_section = estimate_tokens_per_section(sections)
total_tokens = estimate_total_tokens(sections)
print("Tokens per section:", tokens_per_section)
print("Total tokens:", total_tokens)

print("\n--- EMBEDDING GENERATION ---")
embeddings = get_embeddings([query] + sections[:2])  # Just show for first 2 sections for brevity
print("Embedding for query (first 5 dims):", embeddings[0][:5])
print("Embedding for section 1 (first 5 dims):", embeddings[1][:5])

print("\n--- RELEVANCE RANKING ---")
ranked = rank_segments_by_relevance(sections, query, get_embeddings)
for i, (seg, score) in enumerate(ranked[:3]):
    print(f"Top {i+1} segment (score={score:.3f}): {seg[:60]}...")

print("\n--- PARAPHRASING (LLM COMPRESSION) ---")
short_prompt = " ".join(sections[:3])
paraphrased = paraphrase_prompt(short_prompt, instructions="Compress and keep all key info.", max_tokens=40)
print("Paraphrased prompt:", paraphrased)

print("\n--- END-TO-END OPTIMIZATION ---")
optimized = optimize_prompt(prompt, query, max_tokens=60)
print("Optimized prompt:\n", optimized) 
>>>>>>> c67a017189c7a090d3d2e663fbc668db3ee8b308
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
<<<<<<<

*Built for the next generation of GenAI and LLM developers. Optimize your prompts, maximize your results!* 
=======

>>>>>>> 
