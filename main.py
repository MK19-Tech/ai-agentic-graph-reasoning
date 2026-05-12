"""
main.py — Entry point for the Agentic Graph Reasoning Engine.
"""

# ── Must be first — suppress LangGraph deprecation warning ───────────────────
import warnings
warnings.filterwarnings("ignore", message=r".*allowed_objects.*")

# ── Load .env BEFORE any local imports ───────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

import logging
import os
import sys
from pathlib import Path

from graph.builder import build_graph

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REPORT_PATH = Path("research_report.md")

# ── Required env vars ─────────────────────────────────────────────────────────
REQUIRED_ENV_VARS = ["GROQ_API_KEY"]


def validate_env() -> None:
    """Abort early if required environment variables are missing."""
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        logger.error(
            "Missing required environment variable(s): %s. "
            "Copy .env.example → .env and fill in the values.",
            ", ".join(missing),
        )
        sys.exit(1)


def save_report(report: str) -> None:
    """Persist the final report to disk."""
    REPORT_PATH.write_text(report, encoding="utf-8")
    logger.info("Report saved → %s", REPORT_PATH.resolve())


def main() -> None:
    validate_env()

    print("\n--- Starting Agentic Graph Reasoning Engine ---")

    try:
        graph = build_graph()
    except Exception as exc:
        logger.error("Failed to build graph: %s", exc, exc_info=True)
        sys.exit(1)

    topic = input("\nEnter research topic: ").strip()
    if not topic:
        logger.error("Topic cannot be empty.")
        sys.exit(1)

    initial_state: dict[str, str] = {
        "topic": topic,
        "plan": "",
        "research_data": "",
        "final_report": "",
    }

    logger.info("Running graph for topic: %r", topic)

    try:
        result = graph.invoke(initial_state)
    except Exception as exc:
        logger.error("Graph execution failed: %s", exc, exc_info=True)
        sys.exit(1)

    final_report: str = result.get("final_report") or "No report generated."
    save_report(final_report)
    print(f"\n✅ Done! Report saved to '{REPORT_PATH}'")


if __name__ == "__main__":
    main()
