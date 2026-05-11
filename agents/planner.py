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
You are a senior AI research planner.

Create a concise research plan for:

{topic}

Return:
1. Objectives
2. Key research areas
3. Suggested search directions
4. Expected outcomes
"""

        response = llm.invoke(
            [HumanMessage(content=prompt)]
        )

        plan = response.content

        return {
            **state,
            "plan": plan
        }

    except Exception as e:

        logger.error(
            f"Error in planner_node: {e}"
        )

        return {
            **state,
            "plan": f"Planner failed: {str(e)}"
        }
