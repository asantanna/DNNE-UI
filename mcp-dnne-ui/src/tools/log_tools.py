"""Log analysis and management tools for DNNE UI MCP Server"""

import asyncio
import logging
import re
from typing import Dict, Any, List, Optional
try:
    from ..utils.helpers import format_mcp_response, parse_log_metrics
    from ..utils.selectors import *
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.helpers import format_mcp_response, parse_log_metrics
    from utils.selectors import *

logger = logging.getLogger(__name__)

class LogTools:
    """Tools for log analysis and management in DNNE UI"""
    
    def __init__(self, browser_controller, state: Dict[str, Any]):
        """
        Initialize log tools
        
        Args:
            browser_controller: BrowserController instance
            state: Shared state dictionary
        """
        self.browser = browser_controller
        self.state = state
    
    async def get_client_logs(self, client_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get logs for a specific client or current selection
        
        Args:
            client_name: Optional client name to get logs for
        
        Returns:
            MCP response with log content
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Getting logs for client: {client_name or 'current'}")
            
            # Select client if specified
            if client_name:
                from .client_tools import ClientTools
                client_tools = ClientTools(self.browser, self.state)
                result = await client_tools.select_client(client_name)
                if not result["success"]:
                    return result
            
            # Get log content
            log_text = await self.browser.evaluate("""
                () => {
                    // Try various selectors for log content
                    const logPanel = document.querySelector('.log-panel, .logs-container, .log-content, [class*="log"]');
                    if (logPanel) {
                        return logPanel.textContent || logPanel.innerText || '';
                    }
                    
                    // Try pre or code elements that might contain logs
                    const preElement = document.querySelector('pre.logs, pre[class*="log"]');
                    if (preElement) {
                        return preElement.textContent;
                    }
                    
                    // Try textarea that might contain logs
                    const textarea = document.querySelector('textarea[readonly], textarea.logs');
                    if (textarea) {
                        return textarea.value;
                    }
                    
                    return null;
                }
            """)
            
            if log_text:
                # Count errors and warnings
                error_count = len(re.findall(r'\b(ERROR|Error|error)\b', log_text))
                warning_count = len(re.findall(r'\b(WARNING|Warning|warning|WARN|Warn)\b', log_text))
                
                return format_mcp_response(
                    True,
                    data={
                        "client": client_name or self.state.get("selected_client", "default"),
                        "logs": log_text,
                        "error_count": error_count,
                        "warning_count": warning_count,
                        "log_length": len(log_text)
                    },
                    message=f"Retrieved {len(log_text)} characters of logs"
                )
            else:
                return format_mcp_response(
                    True,
                    data={
                        "client": client_name or self.state.get("selected_client", "default"),
                        "logs": "",
                        "error_count": 0,
                        "warning_count": 0,
                        "log_length": 0
                    },
                    message="No logs found"
                )
                
        except Exception as e:
            logger.error(f"Failed to get client logs: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def get_training_metrics(self) -> Dict[str, Any]:
        """
        Extract training metrics from logs
        
        Returns:
            MCP response with parsed metrics
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Extracting training metrics from logs")
            
            # First get the logs
            log_result = await self.get_client_logs()
            
            if not log_result["success"]:
                return log_result
            
            log_text = log_result.get("logs", "")
            
            if not log_text:
                return format_mcp_response(
                    True,
                    data={"metrics": {}, "found": False},
                    message="No logs to parse"
                )
            
            # Parse metrics using helper function
            metrics = parse_log_metrics(log_text)
            
            # Look for additional metrics
            # Training progress
            progress_match = re.search(r'(\d+)/(\d+)', log_text)
            if progress_match:
                metrics["current_batch"] = int(progress_match.group(1))
                metrics["total_batches"] = int(progress_match.group(2))
            
            # Time metrics
            time_match = re.search(r'Time[:\s]+([\d.]+)\s*(ms|s)', log_text, re.IGNORECASE)
            if time_match:
                time_value = float(time_match.group(1))
                if time_match.group(2) == 's':
                    time_value *= 1000  # Convert to ms
                metrics["time_ms"] = time_value
            
            # Check if we found any metrics
            found = any(v is not None for v in metrics.values())
            
            return format_mcp_response(
                True,
                data={
                    "metrics": metrics,
                    "found": found,
                    "epoch": metrics.get("epoch"),
                    "loss": metrics.get("loss"),
                    "accuracy": metrics.get("accuracy")
                },
                message=f"Extracted {sum(1 for v in metrics.values() if v is not None)} metrics"
            )
            
        except Exception as e:
            logger.error(f"Failed to get training metrics: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def get_export_errors(self) -> Dict[str, Any]:
        """
        Find export-related errors in logs
        
        Returns:
            MCP response with export errors
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Searching for export errors")
            
            # Get logs
            log_result = await self.get_client_logs()
            
            if not log_result["success"]:
                return log_result
            
            log_text = log_result.get("logs", "")
            
            if not log_text:
                return format_mcp_response(
                    True,
                    data={"errors": [], "count": 0},
                    message="No logs to search"
                )
            
            errors = []
            
            # Common export error patterns
            error_patterns = [
                r'(Export failed[:\s].*)',
                r'(Failed to export[:\s].*)',
                r'(Export error[:\s].*)',
                r'(Widget mismatch.*)',
                r'(node \d+ has \d+ widget values.*)',
                r'(Missing node[:\s].*)',
                r'(Invalid export.*)',
                r'(Export.*failed.*)',
                r'(Error during export.*)',
                r'(Cannot export.*)'
            ]
            
            for pattern in error_patterns:
                matches = re.finditer(pattern, log_text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    error_text = match.group(1)
                    # Try to get line number or timestamp
                    line_start = max(0, match.start() - 50)
                    context = log_text[line_start:match.start()]
                    
                    # Look for timestamp
                    time_match = re.search(r'(\d{2}:\d{2}:\d{2})', context)
                    timestamp = time_match.group(1) if time_match else None
                    
                    errors.append({
                        "error": error_text,
                        "timestamp": timestamp,
                        "position": match.start()
                    })
            
            # Sort by position (chronological order)
            errors.sort(key=lambda x: x["position"])
            
            # Remove position from final output
            for error in errors:
                del error["position"]
            
            return format_mcp_response(
                True,
                data={
                    "errors": errors,
                    "count": len(errors)
                },
                message=f"Found {len(errors)} export-related errors"
            )
            
        except Exception as e:
            logger.error(f"Failed to get export errors: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def get_recent_errors(self, count: int = 10) -> Dict[str, Any]:
        """
        Get the most recent error messages
        
        Args:
            count: Number of recent errors to return
        
        Returns:
            MCP response with recent errors
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Getting {count} most recent errors")
            
            # Get logs
            log_result = await self.get_client_logs()
            
            if not log_result["success"]:
                return log_result
            
            log_text = log_result.get("logs", "")
            
            if not log_text:
                return format_mcp_response(
                    True,
                    data={"errors": [], "timestamps": []},
                    message="No logs to search"
                )
            
            errors = []
            timestamps = []
            
            # Find all error lines
            error_pattern = r'^.*\b(ERROR|Error|FAIL|Failed|Exception)\b.*$'
            matches = list(re.finditer(error_pattern, log_text, re.MULTILINE | re.IGNORECASE))
            
            # Get the most recent ones (last N matches)
            recent_matches = matches[-count:] if len(matches) > count else matches
            
            for match in recent_matches:
                error_line = match.group(0).strip()
                errors.append(error_line)
                
                # Try to extract timestamp
                time_match = re.search(r'(\d{2}:\d{2}:\d{2})', error_line)
                timestamps.append(time_match.group(1) if time_match else None)
            
            return format_mcp_response(
                True,
                data={
                    "errors": errors,
                    "timestamps": timestamps,
                    "total_errors": len(matches)
                },
                message=f"Found {len(errors)} recent errors (total: {len(matches)})"
            )
            
        except Exception as e:
            logger.error(f"Failed to get recent errors: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def wait_for_log_pattern(self, pattern: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Wait for a specific pattern to appear in logs
        
        Args:
            pattern: Regex pattern to search for
            timeout: Maximum time to wait in seconds
        
        Returns:
            MCP response with match result
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Waiting for pattern: {pattern} (timeout: {timeout}s)")
            
            start_time = asyncio.get_event_loop().time()
            last_log_length = 0
            
            while (asyncio.get_event_loop().time() - start_time) < timeout:
                # Get current logs
                log_result = await self.get_client_logs()
                
                if log_result["success"]:
                    log_text = log_result.get("logs", "")
                    
                    # Check if logs have grown (new content)
                    current_length = len(log_text)
                    if current_length > last_log_length:
                        # Search for pattern in new content
                        new_content = log_text[last_log_length:] if last_log_length > 0 else log_text
                        
                        match = re.search(pattern, new_content, re.IGNORECASE | re.MULTILINE)
                        if match:
                            return format_mcp_response(
                                True,
                                data={
                                    "found": True,
                                    "match": match.group(0),
                                    "timeout": False,
                                    "elapsed_seconds": asyncio.get_event_loop().time() - start_time
                                },
                                message=f"Pattern found: {match.group(0)}"
                            )
                        
                        last_log_length = current_length
                
                # Wait before checking again
                await asyncio.sleep(1)
            
            return format_mcp_response(
                False,
                data={
                    "found": False,
                    "match": None,
                    "timeout": True,
                    "elapsed_seconds": timeout
                },
                error=f"Pattern not found after {timeout} seconds"
            )
            
        except Exception as e:
            logger.error(f"Failed while waiting for log pattern: {e}")
            return format_mcp_response(False, error=str(e))