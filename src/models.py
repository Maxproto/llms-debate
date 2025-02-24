"""
src/debate_agent_lc_models.py

A single place to define create_langchain_llm(...) for each model_name.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain.llms.base import BaseLLM

try:
    from langchain_openai import ChatOpenAI # for GPT-4o, o3-mini
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from langchain_anthropic import ChatAnthropic # for Claude
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from langchain_mistralai import ChatMistralAI # for Mistral
    HAS_MISTRAL = True
except ImportError:
    HAS_MISTRAL = False

try:
    from langchain_ollama import OllamaLLM # for Llama local
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI # for Gemini
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

import os

def create_langchain_llm(model_name: str, temperature: float = 0.0) -> BaseLLM:
    """
    Given a model_name, return an official LangChain LLM or ChatModel instance
    for one of:
      - gpt-4o (OpenAI)
      - o3-mini (OpenAI)
      - claude-3.5-haiku (Anthropic)
      - mistral-small-latest (Mistral)
      - llama-3.2-3b (Ollama local)
      - gemini-2.0-flash (Google Gen AI)

    :param model_name: Name of the desired model (string).
    :param temperature: Float controlling creativity. E.g., 0.0 => deterministic.

    Returns a LangChain LLM or ChatModel instance that can generate text.
    Raises ValueError if the required integration isn't installed or recognized.
    """
    # For GPT-4o / o3-mini via ChatOpenAI
    if model_name == "gpt-4o":
        if not HAS_OPENAI:
            raise ValueError("OpenAI integration not installed. pip install openai langchain")
        return ChatOpenAI(
            model_name="gpt-4o",
            temperature=temperature,
            max_tokens=400
        )
    elif model_name == "o3-mini":
        # Note: o3-mini typically requires a certain temperature (like 1.0),
        if not HAS_OPENAI:
            raise ValueError("OpenAI integration not installed. pip install openai langchain")
        return ChatOpenAI(
            model_name="o3-mini",
            max_tokens=400
        )

    # For Claude 3.5 Haiku via ChatAnthropic
    elif model_name == "claude-3.5-haiku":
        if not HAS_ANTHROPIC:
            raise ValueError("Anthropic integration not installed. pip install anthropic langchain-anthropic")
        return ChatAnthropic(
            model="claude-3-5-haiku-20241022",
            temperature=temperature,
            max_tokens=400
        )

    # For Mistral Small 3 via ChatMistralAI
    elif model_name == "mistral-small-latest":
        if not HAS_MISTRAL:
            raise ValueError("Mistral integration not installed. pip install mistralai langchain-mistralai")
        return ChatMistralAI(
            model="mistral-small-latest",
            temperature=temperature,
            max_tokens=400
        )

    # For Llama 3.2-3B local (Ollama)
    elif model_name == "llama-3.2-3b":
        if not HAS_OLLAMA:
            raise ValueError("Ollama integration not installed. pip install ollama langchain-ollama")
        return OllamaLLM(
            model="llama3.2",
            temperature=temperature,
            base_url="http://localhost:11434",  # default port
        )

    # For Gemini 2.0 Flash (Google Gen AI)
    elif model_name == "gemini-2.0-flash":
        if not HAS_GENAI:
            raise ValueError("Google Gen AI integration not installed. pip install google-cloud-aiplatform langchain_google_genai")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=temperature,
            max_tokens=400,
            google_api_key=os.getenv("GENAI_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported model_name={model_name}. Supported: gpt-4o, o3-mini, claude-3.5-haiku, mistral-small-latest, llama-3.2-3b, gemini-2.0-flash")