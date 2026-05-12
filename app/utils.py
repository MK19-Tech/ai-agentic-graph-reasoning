from functools import wraps
from app.state import logger

def handle_node_errors(func):
    @wraps(func)
    def wrapper(state, *args, **kwargs):
        try:
            return func(state, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return {"error": str(e), "is_sufficient": True}
    return wrapper
