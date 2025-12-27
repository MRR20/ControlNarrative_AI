import os
from dotenv import load_dotenv

def load_mistral_key():
    # Load .env if present
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise RuntimeError(
            "❌ MISTRAL_API_KEY not found.\n"
            "Set it using:\n"
            "export MISTRAL_API_KEY=your_key_here\n"
            "or create a .env file with:\n"
            "MISTRAL_API_KEY=your_key_here"
        )

    os.environ["MISTRAL_API_KEY"] = api_key
    return api_key
