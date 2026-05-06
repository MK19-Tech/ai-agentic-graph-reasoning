import os
from dotenv import load_dotenv
load_dotenv() 

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from app.state import AgentState, logger
from app.nodes import planner_node, executor_node, critic_node, finalizer_node

def router(state: AgentState):
    if state.get("error"): return END
    if state.get("is_sufficient") or state.get("steps_taken", 0) >= len(state.get("plan", [])):
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
    config = {"configurable": {"thread_id": "free_research_01"}}
    task = {"task": "Latest open-source alternatives to OpenAI Sora in 2024"}
    
    print("\n--- Starting Free Agentic Engine ---")
    for event in app.stream(task, config):
        print(f"Node: {list(event.keys())}")

    if input("\nPlan ready. Type 'GO' to start free search: ").upper() == "GO":
        for event in app.stream(None, config):
            print(f"Node: {list(event.keys())}")
        print("\n✅ Done! Report saved to 'research_report.md'.")
