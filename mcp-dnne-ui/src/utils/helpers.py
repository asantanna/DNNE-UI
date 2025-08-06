"""Helper functions for DNNE UI MCP Server"""

import asyncio
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the MCP server"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with optional default"""
    return os.getenv(key, default)

def ensure_screenshot_dir(dir_path: Optional[str] = None) -> Path:
    """Ensure screenshot directory exists in MCP directory"""
    if dir_path is None:
        # Always use the MCP screenshots directory
        mcp_dir = Path(__file__).parent.parent.parent  # Go up to mcp-dnne-ui
        path = mcp_dir / "screenshots"
    else:
        path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path.absolute()

async def retry_with_backoff(
    func, 
    max_retries: int = 3, 
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 30.0,
    retry_on: Optional[tuple] = None
) -> Any:
    """
    Retry a function with exponential backoff
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Factor to multiply delay by after each retry
        max_delay: Maximum delay between retries
        retry_on: Tuple of exception types to retry on (None = all exceptions)
    
    Returns:
        Result of the function call
    
    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            # Check if we should retry on this exception type
            if retry_on and not isinstance(e, retry_on):
                logger.error(f"Non-retryable exception: {e}")
                raise
            
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                logger.info(f"Retrying in {delay:.1f} seconds...")
                await asyncio.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                logger.error(f"All {max_retries} attempts failed. Last error: {e}")
    
    raise last_exception

class RetryableOperation:
    """Context manager for retryable operations with state tracking"""
    
    def __init__(self, operation_name: str, max_retries: int = 3):
        self.operation_name = operation_name
        self.max_retries = max_retries
        self.attempts = 0
        self.errors = []
        
    async def __aenter__(self):
        self.attempts += 1
        logger.debug(f"Starting {self.operation_name} (attempt {self.attempts}/{self.max_retries})")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.errors.append(str(exc_val))
            if self.attempts < self.max_retries:
                logger.warning(f"{self.operation_name} failed: {exc_val}")
                return False  # Don't suppress exception
            else:
                logger.error(f"{self.operation_name} failed after {self.attempts} attempts")
        else:
            logger.debug(f"{self.operation_name} succeeded on attempt {self.attempts}")
        return False

def parse_menu_path(path: str) -> tuple[list[str], str]:
    """
    Parse a menu path like "Workflow/Save As" into components
    
    Args:
        path: Menu path with / separators
    
    Returns:
        Tuple of (menu_items, final_item)
    """
    parts = path.split('/')
    if len(parts) < 2:
        raise ValueError(f"Invalid menu path: {path}")
    
    return parts[:-1], parts[-1]

def extract_error_info(page_content: str) -> Dict[str, Any]:
    """
    Extract error information from page content or dialog
    
    Args:
        page_content: HTML or text content containing error
    
    Returns:
        Dictionary with error details
    """
    # This will be implemented based on actual error format
    return {
        "has_error": False,
        "title": "",
        "message": "",
        "type": ""
    }

def parse_log_metrics(log_text: str) -> Dict[str, Any]:
    """
    Parse training metrics from log text
    
    Args:
        log_text: Log text containing training output
    
    Returns:
        Dictionary with extracted metrics
    """
    import re
    
    metrics = {
        "epoch": None,
        "loss": None,
        "accuracy": None,
        "learning_rate": None
    }
    
    # Parse epoch
    epoch_match = re.search(r'Epoch[:\s]+(\d+)', log_text, re.IGNORECASE)
    if epoch_match:
        metrics["epoch"] = int(epoch_match.group(1))
    
    # Parse loss
    loss_match = re.search(r'Loss[:\s]+([\d.]+)', log_text, re.IGNORECASE)
    if loss_match:
        metrics["loss"] = float(loss_match.group(1))
    
    # Parse accuracy
    acc_match = re.search(r'Accuracy[:\s]+([\d.]+)%?', log_text, re.IGNORECASE)
    if acc_match:
        metrics["accuracy"] = float(acc_match.group(1))
    
    # Parse learning rate
    lr_match = re.search(r'LR[:\s]+([\d.e-]+)', log_text, re.IGNORECASE)
    if lr_match:
        metrics["learning_rate"] = float(lr_match.group(1))
    
    return metrics

def normalize_ui_text(text: str, strip_emojis: bool = True) -> str:
    """
    Normalize UI text by stripping emojis and normalizing whitespace
    
    Args:
        text: Text to normalize
        strip_emojis: Whether to strip common UI emojis (default: True)
    
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Convert to string if needed
    text = str(text)
    
    if strip_emojis:
        # Common UI emojis to remove
        ui_emojis = [
            "📍",  # Location pin (used for Local client)
            "🖥️",  # Computer/desktop (used for remote clients)
            "📂",  # Folder
            "🔄",  # Refresh/reload
            "✅",  # Checkmark/success
            "❌",  # Error/failure
            "⚠️",  # Warning
            "💾",  # Save
            "📋",  # Clipboard
            "🗑️",  # Trash/delete
            "➕",  # Add/plus
            "➖",  # Remove/minus
            "🔍",  # Search
            "📊",  # Chart/graph
            "🎯",  # Target
            "🚀",  # Launch/deploy
            "⏸️",  # Pause
            "▶️",  # Play
            "⏹️",  # Stop
            "🔗",  # Link
            "📡",  # Connection/network
        ]
        
        for emoji in ui_emojis:
            text = text.replace(emoji, "")
    
    # Normalize whitespace - remove extra spaces and trim
    text = " ".join(text.split())
    
    return text.strip()


def format_mcp_response(
    success: bool, 
    data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format a standardized MCP response
    
    Args:
        success: Whether operation succeeded
        data: Optional data to include
        error: Optional error message
        message: Optional status message
    
    Returns:
        Formatted response dictionary
    """
    response = {"success": success}
    
    if message:
        response["message"] = message
    
    if error:
        response["error"] = error
    
    if data:
        response.update(data)
    
    return response