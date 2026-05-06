import os
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from app.state import logger
from app.utils import handle_node_errors

# 100% Free Tools
def get_model():
    # Get a free API key from https://groq.com
    return ChatGroq(model="llama-3.3-70b-specdec", temperature=0)

def get_search_tool():
    return DuckDuckGoSearchRun()

@handle_node_errors
def planner_node(state):
    logger.info("Planner: Creating research steps...")
    model = get_model()
    prompt = f"Plan 3 search steps for: {state['task']}. Return steps only, one per line."
    response = model.invoke(prompt)
    plan_steps = [s.strip() for s in response.content.strip().split("\n") if s.strip()]
    return {"plan": plan_steps, "steps_taken": 0, "error": None}

@handle_node_errors
def executor_node(state):
    idx = state.get("steps_taken", 0)
    query = state["plan"][idx]
    logger.info(f"Executor: Searching DuckDuckGo for '{query}'")
    
    search = get_search_tool()
    results = search.run(query) # DuckDuckGo uses .run()
    return {"context": [f"\nSource ({query}): {str(results)}"], "steps_taken": idx + 1}

@handle_node_errors
def critic_node(state):
    logger.info("Critic: Validating info...")
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
    prompt = f"Write a professional Markdown report for: {state['task']} using: {context}"
    response = model.invoke(prompt)
    with open("research_report.md", "w", encoding="utf-8") as f:
        f.write(response.content)
    return {"current_response": response.content}
