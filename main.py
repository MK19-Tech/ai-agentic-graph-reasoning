import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# --- ULTIMATE WINDOWS-PROOF ENV LOADING ---
root_dir = Path(__file__).parent
possible_files = [".env", ".env.txt", "app/.env"]

found_path = None
for file_name in possible_files:
    test_path = root_dir / file_name
    if test_path.exists():
        load_dotenv(dotenv_path=test_path, override=True)
        found_path = test_path
        if ".txt" in file_name:
            print(f"⚠️ WARNING: Your env file has a .txt extension: {file_name}. Renaming internally...")
        break

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
    key = os.getenv("GROQ_API_KEY")
    
    if not key:
        print("\n--- 🛑 AGENT ENGINE STOPPED ---")
        print(f"I looked in: {root_dir}")
        print("I could not find 'GROQ_API_KEY' in any .env file.")
        print("\nQUICK FIX: Run this command in your terminal then run the script again:")
        print(f"set GROQ_API_KEY=your_key_here")
        sys.exit(1)
    else:
        print(f"✅ SUCCESS: Key loaded (starts with {key[:6]}). Engine starting...")

    config = {"configurable": {"thread_id": "production_test_01"}}
    inputs = {"task": "What are the top 3 agentic design patterns in 2024?"}
    
    print("\n--- Starting Free Agentic Engine ---")
    for event in app.stream(inputs, config):
        print(f"Node: {list(event.keys())}")

    snapshot = app.get_state(config)
    if not snapshot.values.get("error"):
        print("\n[PAUSED] Plan generated.")
        if input("\nType 'GO' to proceed: ").upper() == "GO":
            for event in app.stream(None, config):
                print(f"Node: {list(event.keys())}")
            print("\n✅ Done! Check 'research_report.md'.")
