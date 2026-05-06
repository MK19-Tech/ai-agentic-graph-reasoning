import os
from pathlib import Path
from dotenv import load_dotenv

# --- FAIL-SAFE KEY LOADING ---
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

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
    # DEBUG CHECK
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print(f"❌ DEBUG: Looking for .env at: {env_path.absolute()}")
        print("❌ DEBUG: GROQ_API_KEY not found in Environment.")
    else:
        print(f"✅ DEBUG: Key found (starts with: {key[:6]}...)")

    config = {"configurable": {"thread_id": "final_fix_run"}}
    inputs = {"task": "Recent breakthroughs in Agentic RAG as of May 2026"}
    
    print("\n--- Starting Free Agentic Engine ---")
    for event in app.stream(inputs, config):
        print(f"Node: {list(event.keys())}")

    snapshot = app.get_state(config)
    if snapshot.values.get("error"):
        print(f"\n🛑 FATAL ERROR: {snapshot.values['error']}")
    else:
        print("\n[PAUSED] Plan generated.")
        if input("\nType 'GO' to proceed: ").upper() == "GO":
            for event in app.stream(None, config):
                print(f"Node: {list(event.keys())}")
            print("\n✅ Done! Check 'research_report.md'.")
