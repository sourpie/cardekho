import traceback
from typing import Any, Dict, Optional

def handle_error(error: Exception) -> str:
    """
    Process an exception and return a user-friendly error message.
    
    Args:
        error: The exception to handle
        
    Returns:
        str: User-friendly error message
    """
    error_type = type(error).__name