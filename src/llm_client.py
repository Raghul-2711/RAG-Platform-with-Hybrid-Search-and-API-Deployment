"""
Optional LLM answer-generation layer.

Supports:
    - none     -> retrieval only
    - gemini   -> Google Gemini
    - anthropic -> Anthropic Claude
    - openai   -> OpenAI

The RAG pipeline first retrieves relevant chunks using BM25 + dense
embeddings. Those chunks are then supplied as context to the selected LLM.
"""

from typing import List

from . import config


RAG_PROMPT_TEMPLATE = """You are a helpful assistant for a Retrieval-Augmented Generation (RAG) system.

Answer the question using ONLY the context provided below.

Rules:
1. Use only information contained in the context.
2. Do not invent or assume information that is not present.
3. If the answer cannot be found in the context, say:
   "I don't know based on the provided documents."
4. Give a clear and concise answer.

Context:
{context}

Question:
{question}

Answer:"""


def build_prompt(question: str, contexts: List[str]) -> str:
    """Build the prompt sent to the selected LLM."""
    context_block = "\n\n---\n\n".join(contexts)

    return RAG_PROMPT_TEMPLATE.format(
        context=context_block,
        question=question,
    )


def generate_answer(question: str, contexts: List[str]) -> str:
    """
    Generate an answer using the configured LLM provider.

    Supported providers:
        none
        gemini
        anthropic
        openai
    """

    provider = config.LLM_PROVIDER.lower().strip()

    prompt = build_prompt(question, contexts)

    # Retrieval-only mode
    if provider == "none" or not provider:
        return (
            "[No LLM configured — set LLM_PROVIDER=gemini, "
            "anthropic, or openai to enable generation]"
        )

    # Google Gemini
    if provider == "gemini":
        return _generate_gemini(prompt)

    # Anthropic Claude
    if provider == "anthropic":
        return _generate_anthropic(prompt)

    # OpenAI
    if provider == "openai":
        return _generate_openai(prompt)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def _generate_gemini(prompt: str) -> str:
    """Generate an answer using Google Gemini."""

    from google import genai

    if not config.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not configured. "
            "Add your Gemini API key to the .env file."
        )

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    response = client.models.generate_content(
        model=config.LLM_MODEL,
        contents=prompt,
    )

    if response.text:
        return response.text.strip()

    return "[Gemini returned an empty response]"


def _generate_anthropic(prompt: str) -> str:
    """Generate an answer using Anthropic Claude."""

    import anthropic

    if not config.ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is not configured. "
            "Add your Anthropic API key to the .env file."
        )

    client = anthropic.Anthropic(
        api_key=config.ANTHROPIC_API_KEY
    )

    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return "".join(
        block.text
        for block in response.content
        if hasattr(block, "text")
    ).strip()


def _generate_openai(prompt: str) -> str:
    """Generate an answer using OpenAI."""

    from openai import OpenAI

    if not config.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not configured. "
            "Add your OpenAI API key to the .env file."
        )

    client = OpenAI(
        api_key=config.OPENAI_API_KEY
    )

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=500,
    )

    return response.choices[0].message.content.strip()