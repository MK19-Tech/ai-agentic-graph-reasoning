from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from app.state import AgentState, logger
from app.utils import handle_node_errors

model = ChatOpenAI(model="gpt-4o", temperature=0)
search_tool = TavilySearchResults(max_results=3)

@handle_node_errors
def planner_node(state: AgentState):
    logger.info("Executing Planner Node")
    prompt = f"Plan a 3-step research for: {state['task']}. Return steps only, separated by newlines."
    response = model.invoke(prompt)
    return {"plan": response.content.strip().split("\n"), "steps_taken": 0}

@handle_node_errors
def executor_node(state: AgentState):
    current_step = state['plan'][state['steps_taken']]
    logger.info(f"Executing Search for: {current_step}")
    
    results = search_tool.invoke(current_step)
    return {
        "context": [f"Step {state['steps_taken']}: {str(results)}"],
        "steps_taken": state['steps_taken'] + 1
    }

@handle_node_errors
def critic_node(state: AgentState):
    logger.info("Executing Critic Node")
    # Self-Correction logic
    prompt = f"Does this information fully answer the task: '{state['task']}'? Context: {state['context']}. Reply ONLY 'YES' or 'NO'."
    response = model.invoke(prompt)
    
    is_done = "YES" in response.content.upper()
    return {"is_sufficient": is_done}
