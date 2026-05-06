import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Proper imports from the app package
from app.state import AgentState, logger
from app.nodes import planner_node, executor_node, critic_node, finalizer_node

load_dotenv()

# 1. Define the Routing Logic
def route_after_critic(state: AgentState):
    if state.get("error"):
        return END
    if state["is_sufficient"] or state["steps_taken"] >= len(state["plan"]):
        return "finalizer"
    return "executor"

# 2. Build the Graph
builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_node("critic", critic_node)
builder.add_node("finalizer", finalizer_node)

# 3. Define Edges
builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", "critic")

builder.add_conditional_edges(
    "critic", 
    route_after_critic
)

builder.add_edge("finalizer", END)

# 4. Compile with Persistence & Human-in-the-loop
memory = MemorySaver()
app = builder.compile(
    checkpointer=memory,
    interrupt_before=["executor"] # Pauses here for human approval
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "portfolio_test_001"}}
    user_input = {"task": "What are the key technical differences between Llama 3.1 and GPT-4o?"}
    
    print("\n--- Starting Engine ---")
    
    # Run until the first interrupt (after planner)
    for event in app.stream(user_input, config):
        for node, values in event.items():
            print(f"Finished Node: {node}")

    # Human Interaction Phase
    snapshot = app.get_state(config)
    print("\nPROPOSED PLAN:")
    for i, step in enumerate(snapshot.values.get("plan", []), 1):
        print(f"{i}. {step}")
        
    choice = input("\nType 'GO' to execute plan or 'EXIT' to quit: ").strip().upper()
    
    if choice == "GO":
        # Resume the graph
        for event in app.stream(None, config):
            for node, values in event.items():
                print(f"Finished Node: {node}")
        print("\n✅ Research Complete! Check 'research_report.md'.")
    else:
        print("❌ Execution cancelled.")
