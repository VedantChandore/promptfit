from promptfit import optimize_prompt
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