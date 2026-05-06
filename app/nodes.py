import os
from langchain_groq import ChatGroq
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from app.state import logger
from app.utils import handle_node_errors

def get_model():
    """Defensively fetch the API key and use the updated production model."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is missing. Check your .env file.")
    
    # Production-ready versatile model
    return ChatGroq(
        model="llama-3.3-70b-versatile", 
        api_key=key, 
        temperature=0
    )

@handle_node_errors
def planner_node(state):
    logger.info("Planner: Creating research steps...")
    model = get_model()
    prompt = (
        f"Plan 3 distinct search steps for: {state['task']}. "
        "Return the steps only, one per line, without numbers or bullets."
    )
    response = model.invoke(prompt)
    
    plan_steps = [s.strip() for s in response.content.strip().split("\n") if s.strip()]
    if not plan_steps:
        raise ValueError("LLM returned an empty plan.")
        
    return {"plan": plan_steps, "steps_taken": 0, "error": None}

@handle_node_errors
def executor_node(state):
    if not state.get("plan"):
        raise ValueError("Cannot execute: No plan found in state.")
        
    idx = state.get("steps_taken", 0)
    # Prevent index out of bounds
    if idx >= len(state["plan"]):
        return {"is_sufficient": True}

    query = state["plan"][idx]
    logger.info(f"Executor: Searching DuckDuckGo for '{query}'")
    
    # Use API Wrapper directly for better compatibility with latest ddgs package
    search = DuckDuckGoSearchAPIWrapper()
    
    try:
        # Attempt to search using the specific plan step
        results = search.run(query)
    except Exception as e:
        logger.warning(f"Search failed for specific step. Falling back to general task search. Error: {e}")
        # Fallback to the main task if the plan step is too complex for the search engine
        results = search.run(state['task'])
    
    return {
        "context": [f"\nStep {idx+1} Findings: {results}"], 
        "steps_taken": idx + 1
    }

@handle_node_errors
def critic_node(state):
    logger.info("Critic: Validating info...")
    if not state.get("context"):
        return {"is_sufficient": False}
        
    model = get_model()
    context = " ".join(state["context"])
    prompt = (
        f"As an AI Judge, evaluate if this context fully answers the task: '{state['task']}'.\n"
        f"Context: {context}\n"
        "Reply ONLY with 'YES' or 'NO'."
    )
    response = model.invoke(prompt)
    
    is_done = "YES" in response.content.upper()
    return {"is_sufficient": is_done}

@handle_node_errors
def finalizer_node(state):
    logger.info("Finalizer: Writing report...")
    model = get_model()
    context = " ".join(state.get("context", ["No data gathered."]))
    prompt = (
        f"Write a professional Markdown research report for: {state['task']} "
        f"based on this gathered information: {context}. Include a summary and key findings."
    )
    response = model.invoke(prompt)
    
    # Ensure Unicode encoding for Windows to prevent emojis/special chars from crashing
    with open("research_report.md", "w", encoding="utf-8") as f:
        f.write(response.content)
        
    return {"current_response": response.content}
