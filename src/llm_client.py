"""
Optional LLM answer-generation layer.

Retrieval (BM25 + embeddings) works with zero API keys. This module only
kicks in when a request asks to generate=true and LLM_PROVIDER is set to
"anthropic" or "openai". Otherwise the API returns extractive results
(the retrieved chunks themselves) and skips generation entirely.
"""

from typing import List

from . import config


RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not contained in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, contexts: List[str]) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    return RAG_PROMPT_TEMPLATE.format(context=context_block, question=question)


def generate_answer(question: str, contexts: List[str]) -> str:
    provider = config.LLM_PROVIDER.lower()
    prompt = build_prompt(question, contexts)

    if provider == "none" or not provider:
        return "[No LLM configured — set LLM_PROVIDER=anthropic|openai to enable generation]"

    if provider == "anthropic":
        return _generate_anthropic(prompt)

    if provider == "openai":
        return _generate_openai(prompt)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def _generate_anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if hasattr(block, "text"))


def _generate_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return resp.choices[0].message.content
