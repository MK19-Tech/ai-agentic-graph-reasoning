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


def planner_node(state):

    try:

        logger.info(
            "Planner: Creating research steps..."
        )

        topic = state.get("topic", "")

        prompt = f"""
You are an expert AI research planner.

Create a concise research plan for:

{topic}

Provide:
1. Main objectives
2. Key technical areas
3. Research directions
4. Expected outcomes
"""

        response = llm.invoke(
            [HumanMessage(content=prompt)]
        )

        return {
            **state,
            "plan": response.content
        }

    except Exception as e:

        logger.exception(
            f"Planner node failed: {e}"
        )

        return {
            **state,
            "plan": f"Planner failed: {str(e)}"
        }
