from typing import Optional

try:
    import cohere
except ImportError:
    cohere = None

from .utils import get_cohere_api_key
from .config import COHERE_LLM_MODEL
from .token_budget import estimate_tokens  # ===== CHANGED: import to enforce budget =====


def paraphrase_prompt(prompt: str, instructions: Optional[str] = None, max_tokens: int = 2048) -> str:
    """Rewrite/compress a prompt using Cohere's LLM, ensuring output fits budget."""
    if cohere is None:
        raise ImportError("cohere package is required for paraphrasing.")
    api_key = get_cohere_api_key()
    co = cohere.Client(api_key)
    system_prompt = (
        "Rewrite the following prompt to fit within the token budget, preserving all key instructions and meaning. "
        "Be as concise as possible."
    )
    if instructions:
        system_prompt += f"\nAdditional instructions: {instructions}"
    # ===== CHANGED: Set generate max_tokens to budget =====
    response = co.generate(
        model=COHERE_LLM_MODEL,
        prompt=f"{system_prompt}\n\nPROMPT:\n{prompt}",
        max_tokens=max_tokens,
        temperature=0.2,
        stop_sequences=["\n\n"]
    )
    text = response.generations[0].text.strip()
    # ===== CHANGED: Check and retry if over budget =====
    if estimate_tokens(text) > max_tokens:
        # Recursive retry with stricter instruction
        stricter = f"Ensure output under {max_tokens} tokens. Further compress."  
        return paraphrase_prompt(text, instructions=stricter, max_tokens=max_tokens)
    return text

