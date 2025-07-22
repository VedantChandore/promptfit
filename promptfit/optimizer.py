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
    5. Paraphrase trimmed content (or full prompt) to enforce budget
    """
    # 1. Split
    sections = split_sentences(prompt)

    # 2. Estimate tokens
    tokens_per_section = estimate_tokens_per_section(sections)
    total_tokens= sum(tokens_per_section)
    # If already within budget
    if total_tokens <= max_tokens:
        return prompt

    # 3. Rank by relevance
    ranked_sections = rank_segments_by_relevance(sections, query, get_embeddings)
    sorted_sections = [s for s, _ in ranked_sections]

    # 4. Prune/trim
    pruned_sections = []
    running_total = 0
    for section in sorted_sections:
        sec_tokens = estimate_tokens(section)
        if running_total + sec_tokens > max_tokens:
            continue
        pruned_sections.append(section)
        running_total += sec_tokens
    pruned_prompt = " ".join(pruned_sections)

    # 5. Always paraphrase to enforce budget
    if not pruned_sections:
        # No section fits: paraphrase original prompt
        try:
            pruned_prompt = paraphrase_prompt(
                prompt,
                instructions="Compress as much as possible.",
                max_tokens=max_tokens
            )
        except Exception:
            # Fallback: paraphrase top section
            pruned_prompt = paraphrase_prompt(
                sorted_sections[0],
                instructions="Compress as much as possible.",
                max_tokens=max_tokens
            )
    else:
        # Paraphrase the pruned prompt to fit budget
        paraphrased = paraphrase_prompt(
            pruned_prompt,
            instructions="Preserve all key instructions and meaning.",
            max_tokens=max_tokens
        )
        retries = 0
        while estimate_tokens(paraphrased) > max_tokens and retries < 2:
            paraphrased = paraphrase_prompt(
                paraphrased,
                instructions="Further compress while keeping meaning.",
                max_tokens=max_tokens
            )
            retries += 1
        pruned_prompt = paraphrased

    return pruned_prompt