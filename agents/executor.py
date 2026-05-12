import logging

from utils.search import duckduckgo_search

logger = logging.getLogger(__name__)


def executor_node(state):

    try:

        plan = state.get("plan", "")

        logger.info(
            f"Executor: Searching DuckDuckGo for '{plan}'"
        )

        results = duckduckgo_search(plan)

        formatted_results = "\n\n".join(

            [
                f"""
Title: {r['title']}

URL: {r['url']}

Snippet:
{r['snippet']}
"""
                for r in results
            ]
        )

        return {
            **state,
            "research_data": formatted_results
        }

    except Exception as e:

        logger.error(
            f"Error in executor_node: {e}"
        )

        return {
            **state,
            "research_data": (
                f"Executor failed: {str(e)}"
            )
        }
