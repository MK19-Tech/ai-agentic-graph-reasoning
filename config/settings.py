"""
config/settings.py
------------------
Optimized settings loader with relaxed API key validation and
suppressed LangChain serializer deprecation warning.
"""

import os
import warnings

# ── Suppress LangGraph/LangChain serializer deprecation warning ───────────────
# Remove once the upstream library ships a new default for `allowed_objects`.
warnings.filterwarnings(
    "ignore",
    message=r".*allowed_objects.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*allowed_objects.*",
    # LangChain uses its own warning category; catch both just in case
)

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    # ── Groq ──────────────────────────────────────────────────────────────────
    groq_api_key: str = field(
        default_factory=lambda: os.environ.get("GROQ_API_KEY", "")
    )
    groq_model: str = field(
        default_factory=lambda: os.environ.get("GROQ_MODEL", "llama3-70b-8192")
    )

    # ── Optional extras ───────────────────────────────────────────────────────
    tavily_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("TAVILY_API_KEY")
    )
    langsmith_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("LANGSMITH_API_KEY")
    )
    langchain_tracing_v2: bool = field(
        default_factory=lambda: os.environ.get(
            "LANGCHAIN_TRACING_V2", "false"
        ).lower()
        == "true"
    )

    def validate(self) -> None:
        """
        Validate required settings.

        Groq API keys are issued in several formats:
          • Legacy:  gsk_<40+ alphanum chars>
          • Current: gsk_<variable-length alphanum/underscore string>
          • Some keys issued via proxies/enterprise gateways have different
            prefixes entirely.

        We therefore only enforce:
          1. The key is non-empty.
          2. It is a printable ASCII string (no stray whitespace / encoding artefacts).

        If you need stricter enforcement, set GROQ_API_KEY_STRICT=true and the
        old `gsk_` prefix + length check will be applied.
        """
        key = self.groq_api_key.strip()

        if not key:
            raise ValueError(
                "❌ GROQ_API_KEY is not set.\n"
                "   Add it to your .env file:  GROQ_API_KEY=gsk_..."
            )

        if not key.isprintable() or " " in key:
            raise ValueError(
                "❌ GROQ_API_KEY contains invalid characters (whitespace / "
                "non-printable bytes).  Check your .env file for stray spaces."
            )

        # Optional strict mode ────────────────────────────────────────────────
        if os.environ.get("GROQ_API_KEY_STRICT", "false").lower() == "true":
            import re
            if not re.match(r"^gsk_[A-Za-z0-9_]{20,}$", key):
                raise ValueError(
                    "❌ GROQ_API_KEY does not match the expected Groq format "
                    "(gsk_<20+ alphanumeric/underscore chars>).\n"
                    "   Disable strict mode by removing GROQ_API_KEY_STRICT "
                    "from your environment."
                )

        # Write back the stripped key so callers always get a clean value
        self.groq_api_key = key


# Singleton – imported everywhere as `from config.settings import settings`
settings = Settings()
settings.validate()
