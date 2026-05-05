from functools import wraps
from app.state import logger

def handle_node_errors(func):
    """Decorator to catch node failures and log them into the state."""
    @wraps(func)
    def wrapper(state, *args, **kwargs):
        try:
            return func(state, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error in node {func.__name__}: {str(e)}")
            return {"error": str(e), "is_sufficient": True} # Kill loop on fatal error
    return wrapper
