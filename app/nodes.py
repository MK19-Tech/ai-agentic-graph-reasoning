from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from app.state import logger
from app.utils import handle_node_errors

# Initialize models and tools
model = ChatOpenAI(model="gpt-4o", temperature=0)
search_tool = TavilySearchResults(max_results=3)

@handle_node_errors
def planner_node(state):
    logger.info("--- EXECUTING PLANNER ---")
    prompt = f"Plan a 3-step research path for: {state['task']}. Return only the steps, one per line."
    response = model.invoke(prompt)
    plan_steps = response.content.strip().split("\n")
    return {"plan": plan_steps, "steps_taken": 0}

@handle_node_errors
def executor_node(state):
    logger.info("--- EXECUTING EXECUTOR (SEARCH) ---")
    # Get the current step based on how many we've done
    current_step_index = state.get("steps_taken", 0)
    step_query = state["plan"][current_step_index]
    
    results = search_tool.invoke(step_query)
    return {
        "context": [f"Findings for {step_query}: {str(results)}"],
        "steps_taken": current_step_index + 1
    }

@handle_node_errors
def critic_node(state):
    logger.info("--- EXECUTING CRITIC ---")
    context_str = " ".join(state["context"])
    prompt = f"Task: {state['task']}\nContext: {context_str}\nIs this enough to answer the task fully? Reply ONLY 'YES' or 'NO'."
    response = model.invoke(prompt)
    
    is_done = "YES" in response.content.upper()
    return {"is_sufficient": is_done}

@handle_node_errors
def finalizer_node(state):
    logger.info("--- EXECUTING FINALIZER ---")
    context_str = " ".join(state["context"])
    prompt = f"Write a professional Markdown report for: {state['task']} using this context: {context_str}"
    response = model.invoke(prompt)
    
    with open("research_report.md", "w") as f:
        f.write(response.content)
        
    return {"current_response": response.content}
