import os
from dotenv import load_dotenv

load_dotenv()


def get_required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Required environment variable '{name}' is not configured."
        )

    return value


GROQ_API_KEY = get_required_env("GROQ_API_KEY")
TAVILY_API_KEY = get_required_env("TAVILY_API_KEY")

MODEL_NAME = "openai/gpt-oss-20b"

LANGCHAIN_TRACING_V2 = (
    os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
)

LANGCHAIN_PROJECT = os.getenv(
    "LANGCHAIN_PROJECT",
    "grounded-research-agent",
)