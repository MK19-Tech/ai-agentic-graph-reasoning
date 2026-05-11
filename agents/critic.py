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
You are an AI research critic.

Validate and summarize the following research.

Research:
{research_data}

Provide:
1. Key Findings
2. Reliability Assessment
3. Technical Insights
4. Final Summary
"""

        response = llm.invoke(
            [HumanMessage(content=prompt)]
        )

        return {
            **state,
            "final_report": response.content
        }

    except Exception as e:

        logger.error(
            f"Error in critic_node: {e}"
        )

        return {
            **state,
            "final_report": (
                f"Critic failed: {str(e)}"
            )
        }
