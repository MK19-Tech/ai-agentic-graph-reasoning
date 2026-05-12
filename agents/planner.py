import logging

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from config.settings import settings

logger = logging.getLogger(__name__)


def planner_node(state):
    # Instantiate inside the function so it always uses the fully loaded key
    llm = ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
    )

    try:
        logger.info("Planner: Creating research steps...")

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

        response = llm.invoke([HumanMessage(content=prompt)])

        return {**state, "plan": response.content}

    except Exception as e:
        logger.error(f"Planner node failed: {e}", exc_info=True)
        return {**state, "plan": f"Planner failed: {str(e)}"}
