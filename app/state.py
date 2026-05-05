import logging
from typing import Annotated, List, TypedDict, Optional
import operator

# Structured Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgentEngine")

class AgentState(TypedDict):
    task: str
    plan: List[str]
    context: Annotated[List[str], operator.add] # Append-only for history
    steps_taken: int
    is_sufficient: bool
    error: Optional[str]
    current_response: str
