import logging
from typing import Annotated, List, TypedDict, Optional
import operator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgentEngine")

class AgentState(TypedDict):
    task: str
    plan: List[str]
    context: Annotated[List[str], operator.add]
    steps_taken: int
    is_sufficient: bool
    error: Optional[str]
    current_response: str
