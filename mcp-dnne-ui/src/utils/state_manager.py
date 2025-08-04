"""State management and persistence for DNNE UI MCP Server"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class StateManager:
    """Manage and persist MCP server state"""
    
    def __init__(self, state_file: str = "mcp_state.json"):
        """
        Initialize state manager
        
        Args:
            state_file: Path to state persistence file
        """
        self.state_file = Path(state_file)
        self.state: Dict[str, Any] = self._load_state()
        self.last_save = datetime.now()
        
    def _load_state(self) -> Dict[str, Any]:
        """Load state from file or create default"""
        default_state = {
            "current_workflow": None,
            "selected_client": None,
            "export_target": "Local",
            "sidebar_open": False,
            "sidebar_tab": None,
            "last_error": None,
            "links_visible": True,
            "session_start": datetime.now().isoformat(),
            "operations_count": 0,
            "last_operation": None
        }
        
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    loaded_state = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    default_state.update(loaded_state)
                    logger.info(f"Loaded state from {self.state_file}")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}. Using defaults.")
        
        return default_state
    
    def save_state(self, force: bool = False) -> bool:
        """
        Save current state to file
        
        Args:
            force: Force save even if recently saved
            
        Returns:
            True if saved successfully
        """
        try:
            # Only save every 30 seconds unless forced
            if not force:
                elapsed = (datetime.now() - self.last_save).total_seconds()
                if elapsed < 30:
                    return True
            
            # Prepare state for JSON serialization
            save_data = self.state.copy()
            save_data["last_saved"] = datetime.now().isoformat()
            
            # Write to temp file first, then rename (atomic operation)
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            # Atomic rename
            temp_file.replace(self.state_file)
            
            self.last_save = datetime.now()
            logger.debug(f"State saved to {self.state_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    def update(self, key: str, value: Any, save: bool = True) -> None:
        """
        Update a state value
        
        Args:
            key: State key to update
            value: New value
            save: Whether to persist immediately
        """
        old_value = self.state.get(key)
        self.state[key] = value
        
        if old_value != value:
            logger.debug(f"State updated: {key} = {value}")
            
        if save:
            self.save_state()
    
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
        self.update(key, new_value, save=False)
        return new_value
    
    def record_operation(self, operation: str, success: bool = True) -> None:
        """
        Record an operation in state
        
        Args:
            operation: Name of the operation
            success: Whether operation succeeded
        """
        self.increment_counter("operations_count")
        self.update("last_operation", {
            "name": operation,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }, save=False)
        
        if not success:
            self.update("last_error", operation, save=False)
        
        self.save_state()
    
    def clear_session_state(self) -> None:
        """Clear session-specific state while preserving configuration"""
        preserve_keys = ["export_target", "links_visible"]
        preserved = {k: v for k, v in self.state.items() if k in preserve_keys}
        
        self.state = self._load_state()
        self.state.update(preserved)
        self.state["session_start"] = datetime.now().isoformat()
        self.state["operations_count"] = 0
        
        self.save_state(force=True)
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


class StateRecovery:
    """Handle state recovery after crashes or restarts"""
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        
    async def recover_ui_state(self, browser_controller) -> Dict[str, Any]:
        """
        Attempt to recover UI to match saved state
        
        Args:
            browser_controller: Browser controller instance
            
        Returns:
            Recovery results
        """
        results = {
            "recovered": [],
            "failed": [],
            "skipped": []
        }
        
        if not browser_controller or not browser_controller.page:
            results["failed"].append("Browser not available")
            return results
        
        state = self.state_manager.state
        
        # Recover sidebar state
        if state.get("sidebar_open") and state.get("sidebar_tab"):
            try:
                tab = state["sidebar_tab"]
                selector = f".{tab}-tab-button"
                await browser_controller.click(selector)
                results["recovered"].append(f"Sidebar tab: {tab}")
            except Exception as e:
                results["failed"].append(f"Sidebar recovery: {e}")
        
        # Note: Workflow recovery would require more complex logic
        # to reload the specific workflow file
        
        return results