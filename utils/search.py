import logging

from ddgs import DDGS

logger = logging.getLogger(__name__)


def duckduckgo_search(
    query: str,
    max_results: int = 5
):

    try:

        results = []

        with DDGS() as ddgs:

            search_results = ddgs.text(
                query,
                max_results=max_results
            )

            for r in search_results:

                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    }
                )

        return results

    except Exception as e:

        logger.error(
            f"DuckDuckGo Search Error: {e}"
        )

        return [
            {
                "title": "Search Failed",
                "url": "",
                "snippet": str(e)
            }
        ]
