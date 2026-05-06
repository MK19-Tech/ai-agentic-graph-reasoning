import os
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from app.state import logger
from app.utils import handle_node_errors

def get_model():
    """Defensively fetch the API key to prevent 401 errors."""
    key = os.getenv("GROQ_API_KEY")
    if not key or key.startswith("gsk_your"):
        raise ValueError("CRITICAL: GROQ_API_KEY is invalid or missing in .env")
    return ChatGroq(model="llama-3.3-70b-specdec", groq_api_key=key, temperature=0)

@handle_node_errors
def planner_node(state):
    logger.info("Planner: Creating research steps...")
    model = get_model() # Key check happens here
    prompt = f"Plan 3 search steps for: {state['task']}. Return steps only, one per line."
    response = model.invoke(prompt)
    
    plan_steps = [s.strip() for s in response.content.strip().split("\n") if s.strip()]
    if not plan_steps:
        raise ValueError("LLM returned an empty plan.")
        
    return {"plan": plan_steps, "steps_taken": 0, "error": None}

@handle_node_errors
def executor_node(state):
    # Guard: If planner failed, don't try to access 'plan'
    if not state.get("plan"):
        raise ValueError("Cannot execute: No plan found in state. Check previous node errors.")
        
    idx = state.get("steps_taken", 0)
    query = state["plan"][idx]
    logger.info(f"Executor: Searching DuckDuckGo for '{query}'")
    
    search = DuckDuckGoSearchRun()
    results = search.run(query)
    return {"context": [f"\nSource: {results}"], "steps_taken": idx + 1}

@handle_node_errors
def critic_node(state):
    logger.info("Critic: Validating info...")
    if not state.get("context"):
        return {"is_sufficient": False}
        
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
    
    with open("research_report.md", "w", encoding="utf-8") as f:
        f.write(response.content)
    return {"current_response": response.content}
