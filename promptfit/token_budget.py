import re
from typing import List, Dict, Optional

# If Cohere's token estimator is available, import and use it. Otherwise, fallback.
try:
    import cohere
    _has_cohere = True
except ImportError:
    _has_cohere = False

# Fallback: 1 token ≈ 0.75 words (rough estimate)
def estimate_tokens(text: str) -> int:
    """Estimate token count for a string. Use Cohere if available, else fallback."""
    if _has_cohere:
        # This requires a Cohere client and API key, so only use if configured
        try:
            from .utils import get_cohere_api_key
            api_key = get_cohere_api_key()
            co = cohere.Client(api_key)
            resp = co.tokenize(text)
            return len(resp.tokens)
        except Exception:
            # Fallback if Cohere call fails
            pass
    # ===== CHANGED: Use \S+ to count tokens more accurately =====
    # old: words = re.findall(r'\w+', text)
    tokens = re.findall(r'\S+', text)  # includes punctuation and splits on whitespace
    return max(1, int(len(tokens) / 0.75))


def estimate_tokens_per_section(sections: List[str]) -> List[int]:
    """Estimate token usage for each section in a list."""
    return [estimate_tokens(section) for section in sections]


def estimate_total_tokens(sections: List[str]) -> int:
    """Estimate total token usage for a list of sections."""
    return sum(estimate_tokens_per_section(sections))


