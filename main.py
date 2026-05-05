import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from app.state import AgentState, logger
from app.nodes import planner_node, executor_node, critic_node

load_dotenv()

# Build Graph
builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_node("critic", critic_node)

# Define Logic Flow
builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", "critic")

def router(state: AgentState):
    if state.get("error"): return END # Recovery: stop if error
    if state["is_sufficient"] or state["steps_taken"] >= len(state["plan"]):
        return END
    return "executor" # Loop back for next step

builder.add_conditional_edges("critic", router)

# Persistence for Recovery
checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "unique_session_001"}}
    initial_state = {"task": "What are the top 3 open-source Agentic frameworks in 2024?"}
    
    logger.info("Starting Agentic Engine...")
    for output in app.stream(initial_state, config):
        print("--- Node Completed ---")
        print(output)
