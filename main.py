import os
from dotenv import load_dotenv

# IMPORTANT: load_dotenv() MUST come before importing nodes
load_dotenv() 

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from app.state import AgentState, logger
from app.nodes import planner_node, executor_node, critic_node, finalizer_node

def router(state: AgentState):
    if state.get("error"): return END
    if state["is_sufficient"] or state["steps_taken"] >= len(state["plan"]):
        return "finalizer"
    return "executor"

builder = StateGraph(AgentState)
builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_node("critic", critic_node)
builder.add_node("finalizer", finalizer_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", "critic")
builder.add_conditional_edges("critic", router)
builder.add_edge("finalizer", END)

app = builder.compile(checkpointer=MemorySaver(), interrupt_before=["executor"])

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    task = {"task": "Current stock sentiment for Nvidia Blackwell chips"}
    
    print("\n--- Starting Agent ---")
    for event in app.stream(task, config):
        print(f"Completed: {list(event.keys())[0]}")

    print("\n[PAUSED] Plan generated. Reviewing...")
    # Resume after user types GO
    if input("\nType 'GO' to proceed: ").upper() == "GO":
        for event in app.stream(None, config):
            print(f"Completed: {list(event.keys())[0]}")
        print("\nDone! Check research_report.md")
