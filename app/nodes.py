import os
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from app.state import logger
from app.utils import handle_node_errors

def get_model():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or "your-key" in api_key:
        raise ValueError("Valid OPENAI_API_KEY not found in environment.")
    return ChatOpenAI(model="gpt-4o", temperature=0)

@handle_node_errors
def planner_node(state):
    logger.info("Planner: Creating research steps...")
    model = get_model()
    prompt = f"Plan 3 search steps for: {state['task']}. Return steps only, one per line."
    response = model.invoke(prompt)
    
    plan_steps = [s.strip() for s in response.content.strip().split("\n") if s.strip()]
    if not plan_steps:
        raise ValueError("Planner failed to generate a valid list of steps.")
        
    return {"plan": plan_steps, "steps_taken": 0, "error": None}

@handle_node_errors
def executor_node(state):
    # GUARD: Prevent crash if plan is missing
    if not state.get("plan"):
        raise ValueError("Executor failed: No plan found in state. Did the planner fail?")
        
    idx = state.get("steps_taken", 0)
    if idx >= len(state["plan"]):
        return {"is_sufficient": True}

    query = state["plan"][idx]
    logger.info(f"Executor: Searching for '{query}'")
    
    search = TavilySearchResults(max_results=3)
    results = search.invoke(query)
    return {"context": [f"\nSource ({query}): {str(results)}"], "steps_taken": idx + 1}

@handle_node_errors
def critic_node(state):
    logger.info("Critic: Validating info...")
    # GUARD: If context is empty because search failed
    if not state.get("context"):
        return {"is_sufficient": False, "error": "No context gathered to validate."}

    model = get_model()
    context = " ".join(state["context"])
    prompt = f"Does this answer '{state['task']}'? Context: {context}. Reply YES or NO."
    response = model.invoke(prompt)
    return {"is_sufficient": "YES" in response.content.upper()}

@handle_node_errors
def finalizer_node(state):
    logger.info("Finalizer: Writing report...")
    model = get_model()
    context = " ".join(state.get("context", ["No data gathered."]))
    prompt = f"Write a Markdown report for: {state['task']} using: {context}"
    response = model.invoke(prompt)
    
    # Final fix: Ensure current_response is set
    with open("research_report.md", "w", encoding="utf-8") as f:
        f.write(response.content)
    return {"current_response": response.content}
