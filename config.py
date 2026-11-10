import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

KEY_NAMES = [
    "TAVILY_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "LLAMA_API_KEY",
    "LLAMA_BASE_URL",
    "LLAMA_MODEL",
]


def get_key(name):
    # returns None if the value is empty or still the placeholder from .env.example
    value = (os.getenv(name) or "").strip()
    if not value or value.lower().startswith("paste_your"):
        return None
    return value


def load_colab_secrets():
    # only does something when running inside Google Colab
    try:
        from google.colab import userdata
    except ImportError:
        return

    for name in KEY_NAMES:
        if get_key(name) is None:
            try:
                value = userdata.get(name)
            except Exception:
                continue
            if value:
                os.environ[name] = value


load_colab_secrets()
