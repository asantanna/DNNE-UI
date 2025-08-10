"""Workflow management tools for DNNE UI MCP Server"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import format_mcp_response
from utils.js_defs import *
from utils.timing_constants import (
    MENU_TIMEOUT, DIALOG_TIMEOUT, ANIMATION_DELAY, 
    DIALOG_SETTLE_DELAY, WORKFLOW_LOAD_DELAY
)

logger = logging.getLogger(__name__)

class WorkflowTools:
    """Tools for managing workflows in DNNE UI"""
    
    def __init__(self, server):
        """
        Initialize workflow tools
        
        Args:
            server: DNNE_UI_MCPServer instance for dynamic browser access
        """
        self.server = server
    
    @property
    def browser(self):
        """Get browser controller dynamically from server"""
        return self.server.browser_controller
    
    async def save_workflow(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Save the current workflow
        
        Args:
            name: Optional name for save-as operation
        
        Returns:
            MCP response with success status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            if name:
                # Save As operation
                logger.info(f"Saving workflow as: {name}")
                
                # Use UI tools to click the menu item
                from .ui_tools import UITools
                ui_tools = UITools(self.server)
                
                # Click Workflow > Save As
                result = await ui_tools.click_menu_item("Workflow/Save As")
                if not result.get("success"):
                    return result
                    
                await asyncio.sleep(DIALOG_SETTLE_DELAY)  # Give more time for dialog to appear
                
                # Wait for save dialog
                dialog_visible = await self.browser.wait_for_selector(DIALOG, timeout=DIALOG_TIMEOUT)
                if not dialog_visible:
                    return format_mcp_response(False, error="Save dialog did not appear")
                
                # Find input field and enter name
                input_selector = f"{DIALOG} input[type='text']"
                await self.browser.type_text(input_selector, name)
                
                # Click Confirm button using JavaScript for reliability
                await self.browser.evaluate("""
                    const buttons = document.querySelectorAll('.p-dialog-content button');
                    for (const button of buttons) {
                        if (button.textContent.trim() === 'Confirm') {
                            button.click();
                            break;
                        }
                    }
                """)
                
                return format_mcp_response(
                    True,
                    data={"saved_as": name},
                    message=f"Workflow saved as: {name}"
                )
            else:
                # Regular save operation
                logger.info("Saving current workflow")
                
                # Use Ctrl+S shortcut
                await self.browser.evaluate("""
                    document.dispatchEvent(new KeyboardEvent('keydown', {
                        key: 's',
                        ctrlKey: true,
                        bubbles: true
                    }));
                """)
                
                await asyncio.sleep(DIALOG_SETTLE_DELAY)
                
                return format_mcp_response(
                    True,
                    message="Workflow saved"
                )
                
        except Exception as e:
            logger.error(f"Failed to save workflow: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def new_blank_workflow(self) -> Dict[str, Any]:
        """
        Create a new blank workflow
        
        Returns:
            MCP response with success status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Creating new blank workflow")
            
            # Check if submenu is already visible
            submenu_visible = await self.browser.is_visible(MENU_SUBMENU)
            
            if not submenu_visible:
                # Open Workflow menu
                menu_selector = get_menu_item_selector(1)
                await self.browser.click(f"{menu_selector} .p-menubar-item-label")
                await asyncio.sleep(ANIMATION_DELAY)
            
            # Click New Blank Workflow (1st item)
            new_selector = get_submenu_item_selector(1)
            await self.browser.click(new_selector)
            await asyncio.sleep(DIALOG_SETTLE_DELAY)
            
            return format_mcp_response(
                True,
                message="New blank workflow created"
            )
            
        except Exception as e:
            logger.error(f"Failed to create new workflow: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def clear_workflow(self) -> Dict[str, Any]:
        """
        Clear the current workflow
        
        Returns:
            MCP response with success status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Clearing current workflow")
            
            # Use the UI tools to click the menu item
            from .ui_tools import UITools
            ui_tools = UITools(self.server)
            
            # Click Edit > Clear Workflow
            result = await ui_tools.click_menu_item("Edit/Clear Workflow")
            if not result.get("success"):
                return result
            
            # Check for confirmation dialog
            dialog_visible = await self.browser.is_visible(DIALOG)
            if dialog_visible:
                # Click confirm button
                confirm_button = f"{DIALOG_FOOTER} button:has-text('Yes'), {DIALOG_FOOTER} button:has-text('Confirm')"
                await self.browser.click(confirm_button)
                await asyncio.sleep(ANIMATION_DELAY)
            
            return format_mcp_response(
                True,
                message="Workflow cleared"
            )
            
        except Exception as e:
            logger.error(f"Failed to clear workflow: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def open_workflow(self) -> Dict[str, Any]:
        """
        Open the workflow file dialog
        
        Returns:
            MCP response with success status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Opening workflow dialog")
            
            # Check if submenu is already visible
            submenu_visible = await self.browser.is_visible(MENU_SUBMENU)
            
            if not submenu_visible:
                # Open Workflow menu
                menu_selector = get_menu_item_selector(1)
                await self.browser.click(f"{menu_selector} .p-menubar-item-label")
                await asyncio.sleep(ANIMATION_DELAY)
            
            # Click Open Workflow (2nd item)
            open_selector = get_submenu_item_selector(2)
            await self.browser.click(open_selector)
            await asyncio.sleep(DIALOG_SETTLE_DELAY)
            
            # Wait for file dialog or sidebar to open
            # This might open a file input or the workflows sidebar
            
            return format_mcp_response(
                True,
                message="Workflow dialog opened"
            )
            
        except Exception as e:
            logger.error(f"Failed to open workflow dialog: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def get_workflow_list(self) -> Dict[str, Any]:
        """
        Get list of available workflows
        
        Returns:
            MCP response with workflow list
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Getting workflow list")
            
            # Check if sidebar is open and open it if needed
            sidebar_visible = await self.browser.is_visible(SIDEBAR_CONTENT_CONTAINER)
            if not sidebar_visible:
                await self.browser.click(WORKFLOWS_TAB)
                await asyncio.sleep(DIALOG_SETTLE_DELAY)
            
            # Get all workflow items
            workflows = await self.browser.evaluate("""
                () => {
                    const items = document.querySelectorAll('li[aria-label*=".json"]');
                    return Array.from(items).map(item => 
                        item.getAttribute('aria-label')
                    );
                }
            """)
            
            return format_mcp_response(
                True,
                data={
                    "workflows": workflows if workflows else [],
                    "count": len(workflows) if workflows else 0
                },
                message=f"Found {len(workflows) if workflows else 0} workflows"
            )
            
        except Exception as e:
            logger.error(f"Failed to get workflow list: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def load_workflow(self, name: str) -> Dict[str, Any]:
        """
        Load a workflow from the sidebar
        
        Args:
            name: Name of the workflow to load (with or without .json)
        
        Returns:
            MCP response with success status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Loading workflow: {name}")
            
            # Ensure sidebar is open - check if workflows tab is selected
            sidebar_visible = await self.browser.is_visible(".workflows-tab-button.side-bar-button-selected")
            if not sidebar_visible:
                await self.browser.click(WORKFLOWS_TAB)
                await asyncio.sleep(DIALOG_SETTLE_DELAY)
            
            # Add .json if not present
            if not name.endswith('.json'):
                name = f"{name}.json"
            
            # Find and click the workflow
            workflow_selector = get_workflow_selector(name)
            
            # Check if workflow exists
            workflow_exists = await self.browser.is_visible(workflow_selector)
            logger.debug(f"Checking selector: {workflow_selector}")
            logger.debug(f"Workflow exists: {workflow_exists}")
            
            if not workflow_exists:
                # Double-check with JavaScript
                js_check = await self.browser.evaluate(f"""
                    () => {{
                        const elem = document.querySelector('{workflow_selector}');
                        return {{
                            found: !!elem,
                            text: elem ? elem.textContent : null
                        }};
                    }}
                """)
                logger.debug(f"JavaScript check: {js_check}")
                
                return format_mcp_response(
                    False,
                    error=f"Workflow '{name}' not found in sidebar"
                )
            
            # Click the workflow
            await self.browser.click(workflow_selector)
            await asyncio.sleep(WORKFLOW_LOAD_DELAY)  # Wait for workflow to load
            
            # Close the sidebar after successfully loading the workflow
            await self.browser.click(WORKFLOWS_TAB)
            await asyncio.sleep(ANIMATION_DELAY)
            
            return format_mcp_response(
                True,
                data={"workflow": name},
                message=f"Loaded workflow: {name}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load workflow: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def get_current_workflow_name(self) -> Dict[str, Any]:
        """
        Get the name of the currently loaded workflow
        
        Returns:
            MCP response with current workflow name
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Getting current workflow name")
            
            # Try to get from active tab
            workflow_name = await self.browser.evaluate("""
                () => {
                    // Look for active workflow tab
                    const activeTab = document.querySelector('.workflow-tabs .active-tab');
                    if (activeTab) {
                        return activeTab.textContent.trim();
                    }
                    
                    // Look in title
                    const title = document.title;
                    if (title && title !== 'ComfyUI') {
                        return title.replace(' - ComfyUI', '');
                    }
                    
                    // Check state
                    if (window.app && window.app.graph && window.app.graph.name) {
                        return window.app.graph.name;
                    }
                    
                    return 'Unsaved Workflow';
                }
            """)
            
            return format_mcp_response(
                True,
                data={"name": workflow_name},
                message=f"Current workflow: {workflow_name}"
            )
            
        except Exception as e:
            logger.error(f"Failed to get current workflow name: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def export_workflow(self, run_after: bool = False, args: str = None) -> Dict[str, Any]:
        """
        Export the current workflow
        
        Args:
            run_after: Whether to run the workflow after exporting
            args: Optional runner arguments (e.g., "--enable-telemetry 10,11 --timeout 30s")
        
        Returns:
            MCP response with export status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            # Import UI tools for new functionality
            from tools.ui_tools import UITools
            from utils.js_defs import (
                EXPORT_MENU_OVERLAY, 
                EXPORT_WITH_ARGS_MENU_ITEM,
                RUNNER_ARGS_DIALOG,
                EXPORT_WITH_ARGS_BUTTON
            )
            ui_tools = UITools(self.server)
            
            # If args are provided, use the Export with Arguments flow
            if args is not None:
                logger.info(f"Exporting workflow with args: {args}")
                
                # 1. Click the export dropdown arrow to open menu
                dropdown_selector = ".p-splitbutton-dropdown"
                dropdown_exists = await self.browser.is_visible(dropdown_selector)
                if not dropdown_exists:
                    return format_mcp_response(
                        False,
                        error="Export dropdown arrow not found"
                    )
                
                await self.browser.click(dropdown_selector)
                await asyncio.sleep(ANIMATION_DELAY)
                
                # 2. Wait for menu and click "Export with Arguments..." menu item
                # Wait for the tieredmenu overlay to appear
                menu_visible = await self.browser.wait_for_selector(EXPORT_MENU_OVERLAY, timeout=2000)
                if not menu_visible:
                    return format_mcp_response(
                        False,
                        error="Export menu did not appear"
                    )
                
                # Click the menu item using the correct selector for tieredmenu
                menu_item_exists = await self.browser.is_visible(EXPORT_WITH_ARGS_MENU_ITEM)
                if not menu_item_exists:
                    return format_mcp_response(
                        False,
                        error="'Export with Arguments...' menu item not found"
                    )
                
                await self.browser.click(EXPORT_WITH_ARGS_MENU_ITEM)
                await asyncio.sleep(DIALOG_SETTLE_DELAY)
                
                # 3. Wait for dialog to be ready
                dialog_visible = await self.browser.is_visible(RUNNER_ARGS_DIALOG)
                if not dialog_visible:
                    return format_mcp_response(
                        False,
                        error="Runner args dialog did not appear"
                    )
                
                # 4. Check if override checkbox is already enabled
                override_state = await ui_tools.get_checkbox_state("runner_args_dlg/override")
                if not override_state["success"]:
                    return format_mcp_response(
                        False,
                        error=f"Failed to get override checkbox state: {override_state.get('error')}"
                    )
                
                # 5. Enable override if not already enabled
                # Data is merged directly into response, not under 'data' key
                if not override_state.get("checked", False):
                    click_result = await ui_tools.click_checkbox("runner_args_dlg/override")
                    if not click_result["success"]:
                        return click_result  # Pass through the error
                    await asyncio.sleep(ANIMATION_DELAY)
                
                # 6. Enter the command line arguments
                text_result = await ui_tools.enter_input_text("runner_args_dlg/cmd_line", args)
                if not text_result["success"]:
                    return text_result  # Pass through the error
                await asyncio.sleep(ANIMATION_DELAY)
                
                # 7. Click "Export with Arguments" button
                export_btn_exists = await self.browser.is_visible(EXPORT_WITH_ARGS_BUTTON)
                if not export_btn_exists:
                    return format_mcp_response(
                        False,
                        error="'Export with Arguments' button not found in dialog"
                    )
                
                await self.browser.click(EXPORT_WITH_ARGS_BUTTON)
                await asyncio.sleep(WORKFLOW_LOAD_DELAY)
                
                return format_mcp_response(
                    True,
                    data={
                        "run_after": run_after,
                        "args": args
                    },
                    message=f"Workflow export initiated with args: {args}"
                )
            
            # Original behavior when no args provided
            from utils.js_snippets import js_is_checkbox_disabled, js_is_checkbox_checked
            from utils.js_defs import RUN_AFTER_EXPORT
            
            # FAIL FAST: Check if checkbox is disabled (Local selected)
            is_disabled = await js_is_checkbox_disabled(self.browser, RUN_AFTER_EXPORT)
            
            if is_disabled:
                # Cannot export with run_after when Local is selected
                return format_mcp_response(
                    False,
                    error="Cannot use 'run after export' with Local client selected. Please select a remote client (e.g., Tardigrade) first."
                )
            
            # Get original checkbox state
            orig_checkbox_state = await js_is_checkbox_checked(self.browser, RUN_AFTER_EXPORT)
            
            # FAIL FAST: If we can't read the checkbox state, something is wrong
            if orig_checkbox_state is None:
                return format_mcp_response(
                    False,
                    error="Failed to read run_after_export checkbox state"
                )
            
            # Determine if we need to temporarily change the checkbox
            must_change_checkbox = (
                (run_after == True and orig_checkbox_state == False) or 
                (run_after == False and orig_checkbox_state == True)
            )
            
            # Temporarily change checkbox if needed
            if must_change_checkbox:
                logger.info(f"Temporarily changing checkbox from {orig_checkbox_state} to {run_after}")
                await self.browser.click(RUN_AFTER_EXPORT)
                await asyncio.sleep(ANIMATION_DELAY)
            
            logger.info(f"Exporting workflow (run_after={run_after})")
            
            # Click export button
            export_button_exists = await self.browser.is_visible(EXPORT_BUTTON)
            if not export_button_exists:
                return format_mcp_response(
                    False,
                    error="Export button not found"
                )
            
            await self.browser.click(EXPORT_BUTTON)
            await asyncio.sleep(WORKFLOW_LOAD_DELAY)  # Wait for export to start
            
            # Restore checkbox to original state if we changed it
            if must_change_checkbox:
                logger.info(f"Restoring checkbox to original state: {orig_checkbox_state}")
                await self.browser.click(RUN_AFTER_EXPORT)
                await asyncio.sleep(ANIMATION_DELAY)
            
            return format_mcp_response(
                True,
                data={"run_after": orig_checkbox_state},
                message=f"Workflow export initiated (run_after: {run_after})"
            )
            
        except Exception as e:
            logger.error(f"Failed to export workflow: {e}")
            return format_mcp_response(False, error=str(e))