"""
Centralized timing constants for DNNE UI MCP Server

This module contains all timeout and delay values used throughout the MCP server,
providing consistent timing behavior and easy maintenance.
"""

# TIMEOUT VALUES (milliseconds)

# Browser lifecycle timeouts
BROWSER_LAUNCH_TIMEOUT = 10000      # 10s - Browser startup
BROWSER_READY_TIMEOUT = 3000        # 3s - UI ready check  
BROWSER_CLOSE_TIMEOUT = 3000        # 3s - Browser shutdown

# UI interaction timeouts (all 1 second for fast, responsive interactions)
SELECTOR_TIMEOUT = 1000             # 1s - Element wait
CLICK_TIMEOUT = 1000                # 1s - Click operations
TYPE_TIMEOUT = 1000                 # 1s - Text input
MENU_TIMEOUT = 1000                 # 1s - Menu navigation
DIALOG_TIMEOUT = 1000               # 1s - Dialog appearance

# Long-running operation timeouts (all 5 seconds)
EXPORT_TIMEOUT = 5000               # 5s - Export operations
LOG_PATTERN_TIMEOUT = 5000          # 5s - Log pattern matching

# Network timeouts (seconds)
HTTP_TIMEOUT = 5                    # 5s - HTTP requests
HEALTH_CHECK_TIMEOUT = 5            # 5s - Health checks

# DELAY VALUES (seconds)

# Universal animation delay (0.3s for all UI animations)
ANIMATION_DELAY = 0.3               # Standard animation wait for all UI operations

# Longer settling delays for complex operations
DIALOG_SETTLE_DELAY = 1.0           # Dialog appearance settling
WORKFLOW_LOAD_DELAY = 2.0           # Workflow loading wait
BROWSER_CLOSE_DELAY = 2.0           # Browser shutdown wait