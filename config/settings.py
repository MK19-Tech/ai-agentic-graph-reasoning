import os
import logging

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Explicitly load .env from project root
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH)


class Settings:

    def __init__(self):

        self.GROQ_API_KEY = (
            os.getenv("GROQ_API_KEY", "")
            .strip()
            .replace('"', "")
            .replace("'", "")
        )

    def validate(self):

        if not self.GROQ_API_KEY:

            raise ValueError(
                "\n❌ GROQ_API_KEY missing in .env\n"
            )

        # More robust validation
        if len(self.GROQ_API_KEY) < 20:

            raise ValueError(
                "\n❌ GROQ_API_KEY seems invalid.\n"
                "Please verify your Groq API key.\n"
            )

        logger.info(
            f"✅ SUCCESS: Key loaded "
            f"(starts with "
            f"{self.GROQ_API_KEY[:8]})"
        )


settings = Settings()

settings.validate()
