import time
from typing import Optional

try:
    import cohere
except ImportError:
    cohere = None

from .utils import get_cohere_api_key
from .config import COHERE_LLM_MODEL
from .token_budget import estimate_tokens


def paraphrase_prompt(prompt: str, instructions: Optional[str] = None, max_tokens: int = 2048) -> str:
    if cohere is None:
        raise ImportError("cohere package is required for paraphrasing.")

    api_key = get_cohere_api_key()
    co = cohere.Client(api_key)

    base_system_prompt = (
        "Rewrite the following prompt to fit within the token budget, preserving all key instructions and meaning. "
        "Be as concise as possible."
    )
    if instructions:
        base_system_prompt += f"\nAdditional instructions: {instructions}"

    retries = 0
    max_retries = 5
    backoff_base = 1
    current_prompt = prompt

    best_attempt = prompt
    best_attempt_tokens = estimate_tokens(prompt)

    while retries <= max_retries:
        try:
            response = co.generate(
                model=COHERE_LLM_MODEL,
                prompt=f"{base_system_prompt}\n\nPROMPT:\n{current_prompt}",
                max_tokens=max_tokens,
                temperature=0.2,
                stop_sequences=["\n\n"]
            )
            text = response.generations[0].text.strip()
            token_count = estimate_tokens(text)

            if token_count < best_attempt_tokens:
                best_attempt = text
                best_attempt_tokens = token_count

            if token_count <= max_tokens:
                return text

            retries += 1
            wait_time = backoff_base * (2 ** (retries - 1))
            print(f"[WARN] Output exceeded {max_tokens} tokens. Retrying in {wait_time}s...")
            time.sleep(wait_time)

            current_prompt = text
            base_system_prompt += f"\nEnsure output under {max_tokens} tokens. Further compress."

        except Exception as e:
            retries += 1
            wait_time = backoff_base * (2 ** (retries - 1))
            print(f"[ERROR] Cohere API error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    print("[INFO] Returning best attempt despite exceeding token limit.")
    return best_attempt
