"""
LLM and embeddings factory.

Picks the right backend (OpenAI or Ollama) based on ``LLM_PROVIDER`` in the
environment.  All other modules should import ``make_llm`` and
``make_embeddings`` from here instead of instantiating models directly, so
swapping providers never requires touching business logic.

Supported providers
-------------------
openai  (default)
    Uses ``ChatOpenAI`` and ``OpenAIEmbeddings``.
    Requires ``OPENAI_API_KEY``.

ollama
    Uses ``ChatOllama`` and ``OllamaEmbeddings`` from ``langchain-ollama``.
    Requires a running Ollama instance at ``OLLAMA_BASE_URL``
    (default: http://localhost:11434).
    Pull the models first:
        ollama pull llama3          # or mistral, gemma2, etc.
        ollama pull nomic-embed-text
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.config import settings


def make_llm(temperature: float | None = None) -> BaseChatModel:
    """Return a chat model for the configured provider."""
    t = temperature if temperature is not None else settings.agent_temperature

    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama  # type: ignore[import]
        return ChatOllama(
            model=settings.ollama_model,
            temperature=t,
            base_url=settings.ollama_base_url,
        )

    # Default: OpenAI
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=t,
        api_key=settings.openai_api_key,
    )


def make_embeddings() -> Embeddings:
    """Return an embeddings model for the configured provider."""
    if settings.llm_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings  # type: ignore[import]
        return OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )

    # Default: OpenAI
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
    )
