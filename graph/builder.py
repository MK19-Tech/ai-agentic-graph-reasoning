from langgraph.graph import END
from langgraph.graph import StateGraph

from agents.critic import critic_node
from agents.executor import executor_node
from agents.planner import planner_node

from graph.state import AgentState


def build_graph():

    workflow = StateGraph(AgentState)

    workflow.add_node(
        "planner",
        planner_node
    )

    workflow.add_node(
        "executor",
        executor_node
    )

    workflow.add_node(
        "critic",
        critic_node
    )

    workflow.set_entry_point(
        "planner"
    )

    workflow.add_edge(
        "planner",
        "executor"
    )

    workflow.add_edge(
        "executor",
        "critic"
    )

    workflow.add_edge(
        "critic",
        END
    )

    return workflow.compile()
