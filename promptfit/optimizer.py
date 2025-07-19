# optimizer.py
# End-to-end logic for pruning and rewriting prompts 

from .token_budget import estimate_tokens, estimate_tokens_per_section, estimate_total_tokens
from .embedder import get_embeddings
from .relevance import rank_segments_by_relevance
from .paraphraser import paraphrase_prompt
from .utils import split_sentences
from .config import DEFAULT_MAX_TOKENS


def optimize_prompt(prompt: str, query: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """
    Optimize a prompt to fit within a token budget:
    1. Split into sentences/sections
    2. Estimate tokens per section
    3. Rank by relevance to query
    4. Prune/trim low-salience sections
    5. Paraphrase if still over budget
    """
    # 1. Split
    sections = split_sentences(prompt)
    if not sections:
        return prompt

    # 2. Estimate tokens
    tokens_per_section = estimate_tokens_per_section(sections)
    total_tokens = sum(tokens_per_section)
    if total_tokens <= max_tokens:
        return prompt

    # 3. Rank by relevance
    ranked = rank_segments_by_relevance(sections, query, get_embeddings)
    sorted_sections = [s for s, _ in ranked]

    # 4. Prune/trim
    pruned_sections = []
    running_total = 0
    for section in sorted_sections:
        section_tokens = estimate_tokens(section)
        if running_total + section_tokens > max_tokens:
            break
        pruned_sections.append(section)
        running_total += section_tokens
    pruned_prompt = " ".join(pruned_sections)

    # 5. Paraphrase if still over budget
    if estimate_tokens(pruned_prompt) > max_tokens:
        pruned_prompt = paraphrase_prompt(pruned_prompt, instructions="Preserve all key instructions and meaning.", max_tokens=max_tokens)

    return pruned_prompt 