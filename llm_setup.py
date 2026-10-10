
import os
from crewai import LLM


MODEL_NAME = "groq/openai/gpt-oss-120b"


def create_llm(api_key=None):
    """Create the Groq LLM used by all agents."""
    key = api_key or os.getenv("GROQ_API_KEY")

    if not key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets."
        )

    return LLM(
        model=MODEL_NAME,
        api_key=key,
        temperature=0.1,
        max_tokens=4000,
    )
