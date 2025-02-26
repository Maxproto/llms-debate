"""
src/models.py

A single place to define create_langchain_llm(...) for each model_name.
Supports an optional max_tokens param for controlling generation length.
"""

import os
from dotenv import load_dotenv
load_dotenv(override=True)
from langchain.llms.base import BaseLLM

try:
    from langchain_openai import ChatOpenAI  # for GPT-4o
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from langchain_anthropic import ChatAnthropic  # for Claude
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from langchain_mistralai import ChatMistralAI  # for Mistral
    HAS_MISTRAL = True
except ImportError:
    HAS_MISTRAL = False

try:
    from langchain_ollama import ChatOllama  # for Llama local
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # for Gemini
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


def create_langchain_llm(
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 400
) -> BaseLLM:
    """
    Create a chat-based LLM or text-based LLM instance for the given model_name.
    Supports a 'max_tokens' param where possible.

    :param model_name: e.g. 'gpt-4o', 'gemini-2.0-flash', etc.
    :param temperature: how creative the model is
    :param max_tokens: maximum tokens per response if supported
    """
    # GPT-4o
    if model_name == "gpt-4o":
        if not HAS_OPENAI:
            raise ValueError("OpenAI integration not installed.")
        return ChatOpenAI(
            model_name="gpt-4o",
            temperature=temperature,
            max_tokens=max_tokens
        )

    # Claude
    elif model_name == "claude-3.5-haiku":
        if not HAS_ANTHROPIC:
            raise ValueError("Anthropic integration not installed.")
        return ChatAnthropic(
            model="claude-3-5-haiku-20241022",
            temperature=temperature,
            max_tokens=max_tokens
        )

    # Mistral
    elif model_name == "mistral-small-latest":
        if not HAS_MISTRAL:
            raise ValueError("Mistral integration not installed.")
        return ChatMistralAI(
            model="mistral-small-latest",
            temperature=temperature,
            max_tokens=max_tokens
        )

    # Llama local
    elif model_name == "llama-3.2-3b":
        if not HAS_OLLAMA:
            raise ValueError("Ollama integration not installed.")
        return ChatOllama(
            model="llama3.2",
            temperature=temperature,
            base_url="http://localhost:11434",
            num_predict=max_tokens
        )

    # Gemini
    elif model_name == "gemini-2.0-flash":
        if not HAS_GENAI:
            raise ValueError("Google GenAI integration not installed.")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=temperature,
            max_tokens=max_tokens,
            google_api_key=os.getenv("GENAI_API_KEY")
        )

    else:
        raise ValueError(f"Unsupported model_name={model_name}. Supported: gpt-4o, "
                         "claude-3.5-haiku, mistral-small-latest, llama-3.2-3b, gemini-2.0-flash")
