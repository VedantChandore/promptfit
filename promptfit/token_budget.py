import re
from typing import List
from functools import lru_cache

try:
    import cohere # type: ignore
    _has_cohere = True
except ImportError:
    _has_cohere = False

_S_NON_WS_RE = re.compile(r'\S+')

@lru_cache(maxsize=4096)
def _estimate_tokens_uncached(text: str) -> int:
    if _has_cohere:
        try:
            from .utils import get_cohere_api_key
            api_key = get_cohere_api_key()
            co = cohere.Client(api_key)
            resp = co.tokenize(text)
            return len(resp.tokens)
        except Exception:
            pass
    tokens = _S_NON_WS_RE.findall(text)
    return max(1, int(len(tokens) / 0.75))

def estimate_tokens(text: str) -> int:
    if text is None:
        return 0
    return _estimate_tokens_uncached(text)

def estimate_tokens_per_section(sections: List[str]) -> List[int]:
    return [estimate_tokens(section) for section in sections]

def estimate_total_tokens(sections: List[str]) -> int:
    return sum(estimate_tokens_per_section(sections))

def clear_token_cache() -> None:
    _estimate_tokens_uncached.cache_clear()
