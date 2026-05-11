import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    @classmethod
    def validate(cls):

        if not cls.GROQ_API_KEY:
            raise ValueError(
                "❌ Missing GROQ_API_KEY in .env"
            )

        if not cls.GROQ_API_KEY.startswith("gsk_"):
            raise ValueError(
                "❌ Invalid Groq API Key format"
            )

        print(
            f"✅ SUCCESS: Key loaded "
            f"(starts with {cls.GROQ_API_KEY[:6]})"
        )


settings = Settings()
settings.validate()
