import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# Always load the project's .env, regardless of the directory used to launch Streamlit.
ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=False)


class LLMFactory:
    """Central registry for the RAG application and research benchmark models."""

    MODELS = {
        "Gemini 2.5 Flash": "gemini-2.5-flash",
        "Gemini 2.5 Flash-Lite": "gemini-2.5-flash-lite",
        "Groq GPT-OSS 120B": "openai/gpt-oss-120b",
        "Groq GPT-OSS 20B": "openai/gpt-oss-20b",
    }

    # Keep this allow-list explicit.  Groq models currently receive text-only
    # prompts in this application, while the Gemini models can accept the
    # LangChain image content blocks used by the multimodal RAG path.
    VISION_MODELS = {
        "Gemini 2.5 Flash",
        "Gemini 2.5 Flash-Lite",
    }

    @staticmethod
    def _google_key():
        # Support both common names so existing .env files continue to work.
        return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    @staticmethod
    def _groq_key():
        return os.getenv("GROQ_API_KEY")

    @classmethod
    def supports_vision(cls, model_name: str) -> bool:
        """Return whether ``model_name`` is safe to use with image payloads."""
        return model_name in cls.VISION_MODELS and model_name in cls.MODELS

    @classmethod
    def model_id(cls, model_name: str):
        """Expose the provider model id without duplicating the registry."""
        return cls.MODELS.get(model_name)

    @staticmethod
    def get_llm(model_name: str):
        model_id = LLMFactory.MODELS.get(model_name)
        if not model_id:
            raise ValueError(f"Unsupported model: {model_name}")

        if model_name.startswith("Gemini"):
            api_key = LLMFactory._google_key()
            if not api_key or api_key.startswith("your_"):
                raise RuntimeError(
                    "Gemini API key is missing. Add GOOGLE_API_KEY or GEMINI_API_KEY "
                    "to the project's .env file."
                )
            return ChatGoogleGenerativeAI(
                model=model_id,
                google_api_key=api_key,
                temperature=0.2,
            )

        api_key = LLMFactory._groq_key()
        if not api_key or api_key.startswith("your_"):
            raise RuntimeError(
                "Groq API key is missing. Add GROQ_API_KEY to the project's .env file."
            )
        return ChatGroq(
            model=model_id,
            groq_api_key=api_key,
            temperature=0.2,
        )

    @staticmethod
    def get_available_models():
        return list(LLMFactory.MODELS.keys())
