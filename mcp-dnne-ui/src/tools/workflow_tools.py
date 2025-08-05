"""Workflow management tools for DNNE UI MCP Server"""

import asyncio
import logging
from typing import Dict, Any, Optional
try:
    from ..utils.helpers import format_mcp_response
    from ..utils.selectors import *
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.helpers import format_mcp_response
    from utils.selectors import *

logger = logging.getLogger(__name__)

class WorkflowTools:
    """Tools for managing workflows in DNNE UI"""
    
    def __init__(self, browser_controller, state: Dict[str, Any]):
        """
        Initialize workflow tools
        
        Args:
            browser_controller: BrowserController instance
            state: Shared state dictionary
        """
        self.browser = browser_controller
        self.state = state
    
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
                
                # Check if submenu is already visible
                submenu_visible = await self.browser.is_visible(MENU_SUBMENU)
                
                if not submenu_visible:
                    # Open Workflow menu
                    menu_selector = get_menu_item_selector(1)  # Workflow is first menu
                    await self.browser.click(f"{menu_selector} .p-menubar-item-label")
                    await asyncio.sleep(0.5)
                
                # Click Save Workflow As (5th item)
                save_as_selector = get_submenu_item_selector(5)
                await self.browser.click(save_as_selector)
                await asyncio.sleep(0.5)
                
                # Wait for save dialog
                dialog_visible = await self.browser.wait_for_selector(DIALOG, timeout=3000)
                if not dialog_visible:
                    return format_mcp_response(False, error="Save dialog did not appear")
                
                # Find input field and enter name
                input_selector = f"{DIALOG} input[type='text']"
                await self.browser.type_text(input_selector, name)
                
                # Click Save button
                save_button = f"{DIALOG_FOOTER} button:has-text('Save')"
                await self.browser.click(save_button)
                
                # Update state
                self.state["current_workflow"] = name
                
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
                
                await asyncio.sleep(1)
                
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
                await asyncio.sleep(0.5)
            
            # Click New Blank Workflow (1st item)
            new_selector = get_submenu_item_selector(1)
            await self.browser.click(new_selector)
            await asyncio.sleep(1)
            
            # Update state
            self.state["current_workflow"] = None
            self.state["sidebar_open"] = False
            
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
            
            # Check if submenu is already visible
            submenu_visible = await self.browser.is_visible(MENU_SUBMENU)
            
            if not submenu_visible:
                # Open Edit menu
                menu_selector = get_menu_item_selector(2)  # Edit is second menu
                await self.browser.click(f"{menu_selector} .p-menubar-item-label")
                await asyncio.sleep(0.5)
            
            # Click Clear Workflow (3rd item in Edit menu)
            clear_selector = get_submenu_item_selector(3)
            await self.browser.click(clear_selector)
            await asyncio.sleep(0.5)
            
            # Check for confirmation dialog
            dialog_visible = await self.browser.is_visible(DIALOG)
            if dialog_visible:
                # Click confirm button
                confirm_button = f"{DIALOG_FOOTER} button:has-text('Yes'), {DIALOG_FOOTER} button:has-text('Confirm')"
                await self.browser.click(confirm_button)
                await asyncio.sleep(0.5)
            
            # Update state
            self.state["current_workflow"] = None
            
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
                await asyncio.sleep(0.5)
            
            # Click Open Workflow (2nd item)
            open_selector = get_submenu_item_selector(2)
            await self.browser.click(open_selector)
            await asyncio.sleep(1)
            
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
            
            # Open workflows sidebar if not open
            if not self.state.get("sidebar_open"):
                await self.browser.click(WORKFLOWS_TAB)
                await asyncio.sleep(1)
                self.state["sidebar_open"] = True
            
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
            
            # Ensure sidebar is open
            sidebar_visible = await self.browser.is_visible(".workflows-tab-button.active")
            if not sidebar_visible:
                await self.browser.click(WORKFLOWS_TAB)
                await asyncio.sleep(1)
            
            # Add .json if not present
            if not name.endswith('.json'):
                name = f"{name}.json"
            
            # Find and click the workflow
            workflow_selector = get_workflow_selector(name)
            
            # Check if workflow exists
            workflow_exists = await self.browser.is_visible(workflow_selector)
            if not workflow_exists:
                return format_mcp_response(
                    False,
                    error=f"Workflow '{name}' not found in sidebar"
                )
            
            # Click the workflow
            await self.browser.click(workflow_selector)
            await asyncio.sleep(2)  # Wait for workflow to load
            
            # Update state
            self.state["current_workflow"] = name
            
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
            
            # Update state
            self.state["current_workflow"] = workflow_name
            
            return format_mcp_response(
                True,
                data={"name": workflow_name},
                message=f"Current workflow: {workflow_name}"
            )
            
        except Exception as e:
            logger.error(f"Failed to get current workflow name: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def export_workflow(self, run_after: bool = False) -> Dict[str, Any]:
        """
        Export the current workflow
        
        Args:
            run_after: Whether to run the workflow after exporting
        
        Returns:
            MCP response with export status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Exporting workflow (run_after={run_after})")
            
            # Click export button
            export_button_exists = await self.browser.is_visible(EXPORT_BUTTON)
            if not export_button_exists:
                return format_mcp_response(
                    False,
                    error="Export button not found"
                )
            
            # Check run after export checkbox if needed
            if run_after:
                checkbox_exists = await self.browser.is_visible(RUN_AFTER_EXPORT)
                if checkbox_exists:
                    # Check if already checked
                    is_checked = await self.browser.evaluate(f"""
                        () => {{
                            const checkbox = document.querySelector('{RUN_AFTER_EXPORT}');
                            return checkbox ? checkbox.checked : false;
                        }}
                    """)
                    
                    if not is_checked:
                        await self.browser.click(RUN_AFTER_EXPORT)
                        await asyncio.sleep(0.3)
            
            # Click export button
            await self.browser.click(EXPORT_BUTTON)
            await asyncio.sleep(2)  # Wait for export to start
            
            # Update state
            self.state["last_export"] = {
                "timestamp": asyncio.get_event_loop().time(),
                "run_after": run_after
            }
            
            return format_mcp_response(
                True,
                data={"run_after": run_after},
                message="Export initiated"
            )
            
        except Exception as e:
            logger.error(f"Failed to export workflow: {e}")
            return format_mcp_response(False, error=str(e))