import logging
from functools import wraps

# 1. Setup the logger so nodes can use it
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgentEngine")

def handle_node_errors(func):
    """Decorator to catch node failures and log them into the state."""
    @wraps(func)
    def wrapper(state, *args, **kwargs):
        try:
            return func(state, *args, **kwargs)
        except Exception as e:
            # 2. Log the actual traceback to your terminal
            logger.error(f"Error in node {func.__name__}: {str(e)}")
            
            # 3. Return a state update that prevents the graph from crashing
            return {
                "error": str(e), 
                "is_sufficient": True, # Stops the loop
                "current_response": f"Critical error in {func.__name__}. Check logs."
            }
    return wrapper
