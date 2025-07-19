# promptoptimizer

A Python library for GenAI and LLM developers to fit rich, multi-section prompts within limited token windows (e.g., Cohere’s command-r-plus, OpenAI, Gemini, Anthropic, etc.).

## Features

- **Token Budget Estimator**: Analyze and estimate token usage for prompt templates and sections.
- **Semantic Relevance Scoring**: Rank prompt segments by importance using Cohere embeddings and cosine similarity.
- **Smart Prompt Pruner**: Drop or trim low-salience sections to fit within token budgets.
- **Paraphrasing Module**: Rewrite over-budget prompts using Cohere’s LLM to preserve meaning and instructions.
- **Modular Design**: Each feature is a standalone module for easy integration.
- **Test-Driven**: Unit tests with mocked or live Cohere API responses.

## Tech Stack

- Python 3.10+
- [Cohere command-r-plus](https://docs.cohere.com/docs/models-overview)
- Embeddings: embed-english-v3.0
- Tokenizer: Cohere’s estimator (or manual)
- Libraries: `cohere`, `scikit-learn`, `tiktoken`, `python-dotenv`, `nltk`/`spacy`, `rich`/`typer`

## Directory Structure

```
promptoptimizer/
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
├── .env
├── requirements.txt
├── README.md
│
└── tests/
    ├── test_token_budget.py
    ├── test_relevance.py
    ├── test_optimizer.py
    └── test_paraphraser.py
```

## Sample Usage

```python
from promptoptimizer import optimize_prompt

query = "Summarize this support ticket with action items."
prompt = """
You are a helpful assistant. This is the customer's full complaint...
<<LONG TEXT>>
Include background, emotional tone, and recommendations.
"""

optimized = optimize_prompt(prompt, query, max_tokens=2048)
print(optimized)
```

## Environment Setup

- Store your `COHERE_API_KEY` in a `.env` file.
- The library loads it securely using `python-dotenv` or `os.environ`. 