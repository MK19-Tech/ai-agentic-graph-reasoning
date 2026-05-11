"""
main.py — Entry point for the Free Agentic Graph Reasoning Engine.
"""

# ── Warning filter must be the very first thing, before any other import ──────
import warnings
warnings.filterwarnings("ignore", message=r".*allowed_objects.*")

import logging
import sys
from pathlib import Path

# ── Load .env before local imports so settings.py sees the keys ───────────────
from dotenv import load_dotenv
load_dotenv()

from graph.builder import build_graph

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REPORT_PATH = Path("research_report.md")


def save_report(report: str) -> None:
    REPORT_PATH.write_text(report, encoding="utf-8")
    logger.info("Report saved → %s", REPORT_PATH.resolve())


def main() -> None:
    print("\n--- Starting Free Agentic Engine ---")

    try:
        graph = build_graph()
    except Exception as exc:
        logger.error("Failed to build graph: %s", exc)
        sys.exit(1)

    topic = input("\nEnter research topic: ").strip()
    if not topic:
        logger.error("Topic cannot be empty.")
        sys.exit(1)

    initial_state = {
        "topic": topic,
        "plan": "",
        "research_data": "",
        "final_report": "",
    }

    logger.info("Running graph for topic: %r", topic)

    try:
        result = graph.invoke(initial_state)
    except Exception as exc:
        logger.error("Graph execution failed: %s", exc)
        sys.exit(1)

    final_report = result.get("final_report") or "No report generated."
    save_report(final_report)

    print(f"\n✅ Done! Report saved to '{REPORT_PATH}'")


if __name__ == "__main__":
    main()
