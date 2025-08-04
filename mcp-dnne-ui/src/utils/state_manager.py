"""State management for DNNE UI MCP Server - In-memory only, no disk persistence"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class StateManager:
    """Manage MCP server state in-memory only (no disk persistence)"""
    
    def __init__(self):
        """Initialize state manager with in-memory state only"""
        self.state: Dict[str, Any] = self._create_default_state()
        logger.info("StateManager initialized with in-memory state only")
    
    def _create_default_state(self) -> Dict[str, Any]:
        """Create default in-memory state"""
        return {
            "session_start": datetime.now().isoformat(),
            "operations_count": 0,
            "last_operation": None,
            "last_error": None,
            # Temporary session state only - will be queried from browser
            "current_workflow": None,
            "selected_client": "Local"
        }
    
    def update(self, key: str, value: Any) -> None:
        """
        Update a state value in memory only
        
        Args:
            key: State key to update
            value: New value
        """
        old_value = self.state.get(key)
        self.state[key] = value
        
        if old_value != value:
            logger.debug(f"State updated: {key} = {value}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a state value
        
        Args:
            key: State key to retrieve
            default: Default value if key not found
            
        Returns:
            State value or default
        """
        return self.state.get(key, default)
    
    def increment_counter(self, key: str = "operations_count") -> int:
        """
        Increment a counter in state
        
        Args:
            key: Counter key to increment
            
        Returns:
            New counter value
        """
        current = self.state.get(key, 0)
        new_value = current + 1
        self.update(key, new_value)
        return new_value
    
    def record_operation(self, operation: str, success: bool = True) -> None:
        """
        Record an operation in memory
        
        Args:
            operation: Name of the operation
            success: Whether operation succeeded
        """
        self.increment_counter("operations_count")
        self.update("last_operation", {
            "name": operation,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        
        if not success:
            self.update("last_error", operation)
    
    def clear_session_state(self) -> None:
        """Reset session state to defaults"""
        self.state = self._create_default_state()
        logger.info("Session state cleared")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for current session"""
        session_start = self.state.get("session_start")
        if session_start:
            try:
                start_time = datetime.fromisoformat(session_start)
                duration = (datetime.now() - start_time).total_seconds()
            except:
                duration = 0
        else:
            duration = 0
        
        return {
            "session_duration_seconds": duration,
            "operations_count": self.state.get("operations_count", 0),
            "last_operation": self.state.get("last_operation"),
            "current_workflow": self.state.get("current_workflow"),
            "selected_client": self.state.get("selected_client")
        }