# ... (existing imports and nodes)
from app.nodes import planner_node, executor_node, critic_node, finalizer_node

builder = StateGraph(AgentState)

# Add all 4 nodes
builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_node("critic", critic_node)
builder.add_node("finalizer", finalizer_node)

# Flow
builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", "critic")

# Conditional Router Logic
def router(state: AgentState):
    if state.get("error"): return END
    if state["is_sufficient"] or state["steps_taken"] >= len(state["plan"]):
        return "finalizer" # Route to synthesis
    return "executor"     # Loop back for more data

builder.add_conditional_edges("critic", router)
builder.add_edge("finalizer", END) # End after report generation

# Compile with HITL
app = builder.compile(checkpointer=MemorySaver(), interrupt_before=["executor"])
