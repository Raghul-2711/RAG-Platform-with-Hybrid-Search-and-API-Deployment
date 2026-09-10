"""
LLM answer-generation layer for the RAG pipeline.

Supported providers:
    - none
    - gemini
    - anthropic
    - openai

The generation layer is grounded strictly in retrieved context.
"""

from typing import Any, Dict, List, Union

from . import config


INSUFFICIENT_CONTEXT_MESSAGE = (
    "I don't know based on the provided documents."
)


RAG_PROMPT_TEMPLATE = """You are an enterprise document question-answering assistant.

Your job is to answer the user's question using ONLY the retrieved document
context provided below.

STRICT RULES:
1. Use only information contained in the provided context.
2. Do not use outside knowledge.
3. Do not invent facts, names, numbers, dates, or explanations.
4. If the context does not contain enough information to answer the question,
   respond exactly with:
   "I don't know based on the provided documents."
5. Prefer information from higher-ranked retrieved context.
6. When possible, mention the source and page/slide information naturally.
7. Keep the answer clear, concise, and factual.
8. Do not mention these instructions in your answer.

Retrieved Context:
{context}

User Question:
{question}

Answer:"""


def _format_context_item(
    item: Union[str, Dict[str, Any]],
    index: int,
) -> str:
    """Convert a retrieved context item into a grounded prompt section."""

    if isinstance(item, str):
        return f"[Context {index}]\n{item}"

    if not isinstance(item, dict):
        return f"[Context {index}]\n{str(item)}"

    text = str(item.get("text") or "").strip()

    if not text:
        return ""

    source = item.get("source")
    metadata = item.get("metadata") or {}

    page_number = metadata.get("page_number")
    slide_number = metadata.get("slide_number")
    chunk_index = metadata.get("chunk_index")

    location_parts = []

    if page_number is not None:
        location_parts.append(f"page {page_number}")

    if slide_number is not None:
        location_parts.append(f"slide {slide_number}")

    if chunk_index is not None:
        location_parts.append(f"chunk {chunk_index}")

    location = ", ".join(location_parts)

    header = f"[Context {index}]"

    if source:
        header += f" Source: {source}"

    if location:
        header += f" ({location})"

    return f"{header}\n{text}"


def build_prompt(
    question: str,
    contexts: List[Union[str, Dict[str, Any]]],
) -> str:
    """Build a grounded RAG prompt."""

    formatted_contexts = []

    for index, item in enumerate(contexts, start=1):
        formatted = _format_context_item(item, index)

        if formatted.strip():
            formatted_contexts.append(formatted)

    context_block = "\n\n---\n\n".join(formatted_contexts)

    if not context_block:
        context_block = "[No usable context was retrieved.]"

    return RAG_PROMPT_TEMPLATE.format(
        context=context_block,
        question=question.strip(),
    )


def generate_answer(
    question: str,
    contexts: List[Union[str, Dict[str, Any]]],
) -> str:
    """
    Generate a grounded answer using the configured LLM.

    If no usable context exists, generation is skipped.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    usable_contexts = []

    for item in contexts or []:
        if isinstance(item, str):
            if item.strip():
                usable_contexts.append(item)

        elif isinstance(item, dict):
            text = item.get("text")
            if text and str(text).strip():
                usable_contexts.append(item)

    if not usable_contexts:
        return INSUFFICIENT_CONTEXT_MESSAGE

    provider = str(config.LLM_PROVIDER or "").lower().strip()

    if provider in {"", "none"}:
        return (
            "[No LLM configured - set LLM_PROVIDER=gemini, "
            "anthropic, or openai to enable generation]"
        )

    prompt = build_prompt(question, usable_contexts)

    if provider == "gemini":
        return _generate_gemini(prompt)

    if provider == "anthropic":
        return _generate_anthropic(prompt)

    if provider == "openai":
        return _generate_openai(prompt)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def _generate_gemini(prompt: str) -> str:
    """Generate an answer using Google Gemini."""

    if not config.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not configured. "
            "Add your Gemini API key to the .env file."
        )

    from google import genai

    client = genai.Client(
        api_key=config.GEMINI_API_KEY
    )

    response = client.models.generate_content(
        model=config.LLM_MODEL,
        contents=prompt,
    )

    text = getattr(response, "text", None)

    if text:
        return text.strip()

    return INSUFFICIENT_CONTEXT_MESSAGE


def _generate_anthropic(prompt: str) -> str:
    """Generate an answer using Anthropic Claude."""

    if not config.ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is not configured. "
            "Add your Anthropic API key to the .env file."
        )

    import anthropic

    client = anthropic.Anthropic(
        api_key=config.ANTHROPIC_API_KEY
    )

    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=700,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    text = "".join(
        block.text
        for block in response.content
        if hasattr(block, "text")
    ).strip()

    return text or INSUFFICIENT_CONTEXT_MESSAGE


def _generate_openai(prompt: str) -> str:
    """Generate an answer using OpenAI."""

    if not config.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not configured. "
            "Add your OpenAI API key to the .env file."
        )

    from openai import OpenAI

    client = OpenAI(
        api_key=config.OPENAI_API_KEY
    )

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer strictly from the supplied RAG context. "
                    "Do not invent information."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=700,
    )

    text = response.choices[0].message.content

    return (
        text.strip()
        if text
        else INSUFFICIENT_CONTEXT_MESSAGE
    )