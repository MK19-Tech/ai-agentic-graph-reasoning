import logging

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from config.settings import settings

logger = logging.getLogger(__name__)

llm = ChatGroq(
    groq_api_key=settings.GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


def critic_node(state):

    try:

        logger.info(
            "Critic: Validating info..."
        )

        research_data = state.get(
            "research_data",
            ""
        )

        prompt = f"""
You are a senior AI research analyst.

Analyze and summarize:

{research_data}

Provide:
1. Key findings
2. Reliability analysis
3. Technical insights
4. Final concise summary
"""

        response = llm.invoke(
            [HumanMessage(content=prompt)]
        )

        return {
            **state,
            "final_report": response.content
        }

    except Exception as e:

        logger.exception(
            f"Critic node failed: {e}"
        )

        return {
            **state,
            "final_report": (
                f"Critic failed: {str(e)}"
            )
        }
