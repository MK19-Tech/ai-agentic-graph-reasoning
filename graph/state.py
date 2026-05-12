from typing import TypedDict


class AgentState(TypedDict):

    topic: str

    plan: str

    research_data: str

    final_report: str
