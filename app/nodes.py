import os
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from app.state import logger
from app.utils import handle_node_errors

# We define a helper to ensure keys are loaded before the model starts
def get_model():
    return ChatOpenAI(model="gpt-4o", temperature=0)

@handle_node_errors
def planner_node(state):
    logger.info("Planner: Creating research steps...")
    model = get_model()
    prompt = f"Plan 3 search steps for: {state['task']}. Return steps only, one per line."
    response = model.invoke(prompt)
    return {"plan": response.content.strip().split("\n"), "steps_taken": 0}

@handle_node_errors
def executor_node(state):
    idx = state.get("steps_taken", 0)
    query = state["plan"][idx]
    logger.info(f"Executor: Searching for '{query}'")
    
    search = TavilySearchResults(max_results=3)
    results = search.invoke(query)
    return {"context": [str(results)], "steps_taken": idx + 1}

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
    context = " ".join(state["context"])
    prompt = f"Write a Markdown report for: {state['task']} using: {context}"
    response = model.invoke(prompt)
    with open("research_report.md", "w", encoding="utf-8") as f:
        f.write(response.content)
    return {"current_response": response.content}
