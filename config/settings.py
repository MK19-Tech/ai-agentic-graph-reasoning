import os
import warnings

# Must be first — suppresses LangGraph serializer warning before any
# langgraph imports happen anywhere in the project.
warnings.filterwarnings("ignore", message=r".*allowed_objects.*")

from dotenv import load_dotenv

load_dotenv()


class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()

    @classmethod
    def validate(cls) -> None:
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "❌ Missing GROQ_API_KEY in .env\n"
                "   Add: GROQ_API_KEY=your_key_here"
            )

        if not cls.GROQ_API_KEY.isprintable() or " " in cls.GROQ_API_KEY:
            raise ValueError(
                "❌ GROQ_API_KEY contains invalid characters. "
                "Check your .env for stray spaces or quotes."
            )

        print(f"✅ Groq API Key loaded (starts with: {cls.GROQ_API_KEY[:6]}...)")


settings = Settings()
settings.validate()
