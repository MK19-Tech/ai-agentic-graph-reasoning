import os
import warnings

warnings.filterwarnings("ignore", message=r".*allowed_objects.*")

from dotenv import load_dotenv

# Load .env immediately when this module is imported
# This guarantees os.getenv() calls below always see the .env values
load_dotenv()


class Settings:
    # Read AFTER load_dotenv() has run above
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()

    @classmethod
    def validate(cls) -> None:
        # Re-read at validation time as a safety net
        cls.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

        if not cls.GROQ_API_KEY:
            raise ValueError(
                "❌ Missing GROQ_API_KEY in .env\n"
                "   Add: GROQ_API_KEY=gsk_your_actual_key_here"
            )

        if not cls.GROQ_API_KEY.isprintable() or " " in cls.GROQ_API_KEY:
            raise ValueError(
                "❌ GROQ_API_KEY has invalid characters. "
                "Check .env for stray spaces or quotes."
            )

        print(f"✅ Groq API Key loaded (starts with: {cls.GROQ_API_KEY[:8]}...)")


settings = Settings()
settings.validate()
