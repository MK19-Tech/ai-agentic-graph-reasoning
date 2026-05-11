import logging

from graph.builder import build_graph


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )
)

logger = logging.getLogger(__name__)


def save_report(report: str):

    with open(
        "research_report.md",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)


def main():

    print(
        "\n--- Starting Free Agentic Engine ---"
    )

    graph = build_graph()

    topic = input(
        "\nEnter research topic: "
    )

    initial_state = {
        "topic": topic,
        "plan": "",
        "research_data": "",
        "final_report": ""
    }

    try:

        result = graph.invoke(initial_state)

        final_report = result.get(
            "final_report",
            "No report generated."
        )

        save_report(final_report)

        print(
            "\n✅ Done! Check "
            "'research_report.md'"
        )

    except Exception as e:

        logger.exception(
            f"Application failed: {e}"
        )


if __name__ == "__main__":
    main()
