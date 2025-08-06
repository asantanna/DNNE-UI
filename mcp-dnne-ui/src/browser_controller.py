"""Browser controller for DNNE UI automation using Playwright"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from playwright.async_api import async_playwright, Browser, Page, Playwright
try:
    from .utils.helpers import ensure_screenshot_dir, retry_with_backoff
    from .utils.timing_constants import (
        BROWSER_LAUNCH_TIMEOUT, BROWSER_READY_TIMEOUT, BROWSER_CLOSE_TIMEOUT,
        SELECTOR_TIMEOUT, CLICK_TIMEOUT, TYPE_TIMEOUT, ANIMATION_DELAY, BROWSER_CLOSE_DELAY
    )
except ImportError:
    from utils.helpers import ensure_screenshot_dir, retry_with_backoff
    from utils.timing_constants import (
        BROWSER_LAUNCH_TIMEOUT, BROWSER_READY_TIMEOUT, BROWSER_CLOSE_TIMEOUT,
        SELECTOR_TIMEOUT, CLICK_TIMEOUT, TYPE_TIMEOUT, ANIMATION_DELAY, BROWSER_CLOSE_DELAY
    )

logger = logging.getLogger(__name__)

class BrowserController:
    """Manages browser instance and page interactions for DNNE UI"""
    
    def __init__(self, dnne_url: str = "http://172.22.160.1:8188", headless: bool = False):
        """
        Initialize browser controller
        
        Args:
            dnne_url: URL of DNNE UI server
            headless: Whether to run browser in headless mode
        """
        self.dnne_url = dnne_url
        self.headless = headless
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.screenshot_dir = ensure_screenshot_dir()
        
    async def initialize(self) -> None:
        """Initialize Playwright and launch browser"""
        logger.info("Initializing browser controller")
        
        self.playwright = await async_playwright().start()
        
        # Launch browser with maximized window for status bar visibility
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--start-maximized',
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage'
            ]
        )
        
        # Create new page with no viewport restrictions
        self.page = await self.browser.new_page(no_viewport=True)
        
        # Navigate to DNNE UI
        await self.navigate_to_dnne()
        
    async def navigate_to_dnne(self) -> bool:
        """
        Navigate to DNNE UI and wait for it to load
        
        Returns:
            True if navigation successful, False otherwise
        """
        if not self.page:
            logger.error("Page not initialized")
            return False
            
        try:
            logger.info(f"Navigating to {self.dnne_url}")
            response = await self.page.goto(
                self.dnne_url,
                wait_until="domcontentloaded",
                timeout=BROWSER_LAUNCH_TIMEOUT
            )
            
            if response and response.status == 200:
                logger.info("Successfully connected to DNNE UI")
                
                # Wait for UI to be ready
                await self.wait_for_ui_ready()
                return True
            else:
                logger.error(f"Failed to connect: status {response.status if response else 'No response'}")
                return False
                
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return False
    
    async def wait_for_ui_ready(self, timeout: int = BROWSER_READY_TIMEOUT) -> bool:
        """
        Wait for DNNE UI to be fully loaded
        
        Args:
            timeout: Maximum wait time in milliseconds
        
        Returns:
            True if UI is ready, False if timeout
        """
        if not self.page:
            return False
            
        try:
            # Wait for key UI elements to be present
            await self.page.wait_for_selector('.side-bar-button', timeout=timeout)
            await self.page.wait_for_selector('.comfyui-menu', timeout=timeout)
            
            # Small delay to ensure everything is rendered
            await asyncio.sleep(ANIMATION_DELAY)
            
            logger.info("UI is ready")
            return True
            
        except Exception as e:
            logger.error(f"UI not ready after {timeout}ms: {e}")
            return False
    
    async def click(self, selector: str, timeout: int = CLICK_TIMEOUT) -> bool:
        """
        Click an element with retry logic
        
        Args:
            selector: CSS selector for element
            timeout: Maximum wait time for element
        
        Returns:
            True if click successful
        """
        if not self.page:
            return False
            
        try:
            element = await self.page.wait_for_selector(selector, timeout=timeout)
            if element:
                await element.click()
                logger.debug(f"Clicked: {selector}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to click {selector}: {e}")
            return False
    
    async def get_text(self, selector: str, timeout: int = SELECTOR_TIMEOUT, normalize: bool = True) -> Optional[str]:
        """
        Get text content of an element
        
        Args:
            selector: CSS selector for element
            timeout: Maximum wait time for element
            normalize: Whether to normalize the text (strip emojis and extra whitespace)
        
        Returns:
            Text content or None if not found
        """
        if not self.page:
            return None
            
        try:
            element = await self.page.wait_for_selector(selector, timeout=timeout)
            if element:
                text = await element.text_content()
                if text and normalize:
                    from utils.helpers import normalize_ui_text
                    return normalize_ui_text(text)
                return text
            return None
            
        except Exception as e:
            logger.error(f"Failed to get text from {selector}: {e}")
            return None
    
    async def type_text(self, selector: str, text: str, timeout: int = TYPE_TIMEOUT) -> bool:
        """
        Type text into an input field
        
        Args:
            selector: CSS selector for input
            text: Text to type
            timeout: Maximum wait time for element
        
        Returns:
            True if successful
        """
        if not self.page:
            return False
            
        try:
            element = await self.page.wait_for_selector(selector, timeout=timeout)
            if element:
                await element.fill(text)
                logger.debug(f"Typed text into: {selector}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to type into {selector}: {e}")
            return False
    
    async def take_screenshot(self, name: str = "screenshot") -> Optional[str]:
        """
        Take a screenshot of the current page
        
        Args:
            name: Name for the screenshot file (without extension)
        
        Returns:
            Path to saved screenshot or None if failed
        """
        if not self.page:
            return None
            
        try:
            filename = self.screenshot_dir / f"{name}.png"
            await self.page.screenshot(path=str(filename))
            logger.info(f"Screenshot saved: {filename}")
            return str(filename)
            
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return None
    
    async def evaluate(self, script: str) -> Any:
        """
        Execute JavaScript in the page context
        
        Args:
            script: JavaScript code to execute
        
        Returns:
            Result of the script execution
        """
        if not self.page:
            return None
            
        try:
            result = await self.page.evaluate(script)
            return result
            
        except Exception as e:
            logger.error(f"Failed to evaluate script: {e}")
            return None
    
    async def wait_for_selector(self, selector: str, timeout: int = SELECTOR_TIMEOUT) -> bool:
        """
        Wait for an element to appear
        
        Args:
            selector: CSS selector to wait for
            timeout: Maximum wait time in milliseconds
        
        Returns:
            True if element found, False if timeout
        """
        if not self.page:
            return False
            
        try:
            await self.page.wait_for_selector(selector, timeout=timeout)
            return True
        except:
            return False
    
    async def is_visible(self, selector: str) -> bool:
        """
        Check if an element is visible
        
        Args:
            selector: CSS selector for element
        
        Returns:
            True if element is visible
        """
        if not self.page:
            return False
            
        try:
            element = await self.page.query_selector(selector)
            if element:
                return await element.is_visible()
            return False
            
        except Exception as e:
            logger.error(f"Failed to check visibility of {selector}: {e}")
            return False
    
    async def get_element_count(self, selector: str) -> int:
        """
        Count elements matching a selector
        
        Args:
            selector: CSS selector
        
        Returns:
            Number of matching elements
        """
        if not self.page:
            return 0
            
        try:
            elements = await self.page.query_selector_all(selector)
            return len(elements)
            
        except Exception as e:
            logger.error(f"Failed to count elements {selector}: {e}")
            return 0
    
    async def cleanup(self) -> None:
        """Clean up browser resources"""
        logger.info("Cleaning up browser controller")
        
        if self.page:
            await self.page.close()
            self.page = None
            
        if self.browser:
            await self.browser.close()
            self.browser = None
            
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
    
    async def restart_browser(self) -> bool:
        """
        Restart the browser (useful for recovery)
        
        Returns:
            True if restart successful
        """
        logger.info("Restarting browser")
        
        try:
            # Save current state if possible
            saved_state = None
            if self.page:
                try:
                    saved_state = await self.page.evaluate("""
                        () => ({
                            url: window.location.href,
                            workflow: document.title
                        })
                    """)
                except:
                    pass
            
            # Clean up existing browser
            await self.cleanup()
            await asyncio.sleep(BROWSER_CLOSE_DELAY)  # Give it time to fully close
            
            # Reinitialize
            await self.initialize()
            
            # Try to restore state
            if saved_state and saved_state.get("url") != self.dnne_url:
                logger.info(f"Restoring previous state: {saved_state}")
                # Navigation happens in initialize, so state should be restored
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart browser: {e}")
            return False
    
    def is_playwright_browser_process_active(self) -> bool:
        """
        Check if Playwright browser process is active
        
        Returns:
            True if browser process exists and is connected
        """
        try:
            return self.browser is not None and self.browser.is_connected()
        except:
            return False
    
    def is_browser_window_available(self) -> bool:
        """
        Check if browser window/page exists and is not closed
        
        Returns:
            True if there is an available browser window
        """
        try:
            return (
                self.page is not None and 
                not self.page.is_closed()
            )
        except:
            return False
    
    async def is_javascript_executable(self) -> bool:
        """
        Check if JavaScript can be executed in the page
        
        Returns:
            True if JavaScript execution is possible
        """
        if not self.is_browser_window_available():
            return False
        
        try:
            result = await self.page.evaluate("() => 1 + 1")
            return result == 2
        except:
            return False
    
    async def is_responsive(self) -> bool:
        """
        High-level check if browser is responsive to operations
        Checks: process active → window available → JS executable
        
        Returns:
            True if browser is fully responsive
        """
        if not self.is_playwright_browser_process_active():
            return False
        if not self.is_browser_window_available():
            return False
        return await self.is_javascript_executable()
    
    async def is_healthy(self) -> bool:
        """
        Check if browser is healthy
        
        Returns:
            True if browser is healthy (after potential restart)
        """
        if await self.is_responsive():
            return True
        
        logger.warning("Browser unhealthy, attempting restart...")
        return await self.restart_browser()
    
    async def handle_unexpected_dialog(self) -> bool:
        """
        Check for and dismiss unexpected dialogs
        
        Returns:
            True if a dialog was dismissed
        """
        if not self.page:
            return False
        
        try:
            dialog_selector = ".p-dialog"
            dialog_visible = await self.page.is_visible(dialog_selector)
            
            if dialog_visible:
                logger.info("Unexpected dialog detected, attempting to dismiss...")
                
                # Try close button first
                close_btn = ".p-dialog-header-close"
                if await self.page.is_visible(close_btn):
                    await self.page.click(close_btn)
                else:
                    # Try Cancel or OK button
                    await self.page.click(".p-dialog-footer button:has-text('Cancel'), .p-dialog-footer button:has-text('OK')")
                
                await asyncio.sleep(ANIMATION_DELAY)
                return True
                
        except Exception as e:
            logger.warning(f"Failed to handle dialog: {e}")
        
        return False
    
    # Browser State Query Methods - Get state directly from browser
    
    async def get_current_workflow(self) -> Optional[str]:
        """
        Get the current workflow name from browser
        
        Returns:
            Workflow name or None
        """
        if not self.page:
            return None
        
        try:
            # Get from document title
            title = await self.page.evaluate("document.title")
            if title and "Unsaved Workflow" not in title:
                return title
            return None
        except Exception as e:
            logger.error(f"Failed to get current workflow: {e}")
            return None
    
    async def get_sidebar_state(self) -> Dict[str, Any]:
        """
        Get current sidebar state from browser
        
        Returns:
            Dict with sidebar_open and active_tab
        """
        if not self.page:
            return {"sidebar_open": False, "active_tab": None}
        
        try:
            state = await self.page.evaluate("""
                () => {
                    const sidebar = document.querySelector('.side-bar-container');
                    const isOpen = sidebar && !sidebar.classList.contains('collapsed');
                    
                    let activeTab = null;
                    if (isOpen) {
                        const activeButton = document.querySelector('.side-bar-button.active');
                        if (activeButton) {
                            if (activeButton.classList.contains('workflows-tab-button')) {
                                activeTab = 'workflows';
                            } else if (activeButton.classList.contains('nodes-tab-button')) {
                                activeTab = 'nodes';
                            }
                        }
                    }
                    
                    return {
                        sidebar_open: isOpen,
                        active_tab: activeTab
                    };
                }
            """)
            return state
        except Exception as e:
            logger.error(f"Failed to get sidebar state: {e}")
            return {"sidebar_open": False, "active_tab": None}
    
    async def get_canvas_info(self) -> Dict[str, Any]:
        """
        Get canvas information from browser
        
        Returns:
            Dict with node_count, zoom_level, links_visible
        """
        if not self.page:
            return {"node_count": 0, "zoom_level": 1.0, "links_visible": True}
        
        try:
            info = await self.page.evaluate("""
                () => {
                    const nodes = document.querySelectorAll('.node');
                    const canvas = document.querySelector('canvas') || document.querySelector('.graph-canvas');
                    
                    // Try to get zoom from transform or data attribute
                    let zoom = 1.0;
                    if (canvas) {
                        const transform = canvas.style.transform;
                        const scaleMatch = transform && transform.match(/scale\\(([^)]+)\\)/);
                        if (scaleMatch) {
                            zoom = parseFloat(scaleMatch[1]);
                        }
                    }
                    
                    // Check if links are visible
                    const links = document.querySelectorAll('.link-line, .connection-line');
                    const linksVisible = links.length > 0 && 
                        (!links[0].style.display || links[0].style.display !== 'none');
                    
                    return {
                        node_count: nodes.length,
                        zoom_level: zoom,
                        links_visible: linksVisible
                    };
                }
            """)
            return info
        except Exception as e:
            logger.error(f"Failed to get canvas info: {e}")
            return {"node_count": 0, "zoom_level": 1.0, "links_visible": True}
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """
        Get agent and client status from status bar
        
        Returns:
            Dict with agent_connected, client_count, selected_client
        """
        if not self.page:
            return {"agent_connected": False, "client_count": 0, "selected_client": "Local"}
        
        try:
            status = await self.page.evaluate("""
                () => {
                    const statusBar = document.querySelector('.agent-status-bar');
                    if (!statusBar) {
                        return {
                            agent_connected: false,
                            client_count: 0,
                            selected_client: 'Local',
                            debug: 'No status bar found'
                        };
                    }
                    
                    const statusText = statusBar.textContent || '';
                    
                    // Parse agent status
                    const agentConnected = statusText.includes('Connected');
                    
                    // Parse client count
                    const clientMatch = statusText.match(/Clients:\\s*(\\d+)/);
                    const clientCount = clientMatch ? parseInt(clientMatch[1]) : 0;
                    
                    // Get selected client from dropdown
                    const clientDropdown = document.querySelector('.export-target-dropdown, .client-dropdown');
                    const selectedClient = clientDropdown ? 
                        (clientDropdown.value || clientDropdown.textContent || 'Local') : 'Local';
                    
                    return {
                        agent_connected: agentConnected,
                        client_count: clientCount,
                        selected_client: selectedClient,
                        debug_status_text: statusText  // Add debug info
                    };
                }
            """)
            return status
        except Exception as e:
            logger.error(f"Failed to get agent status: {e}")
            return {"agent_connected": False, "client_count": 0, "selected_client": "Local"}
    
    async def get_ui_state(self) -> Dict[str, Any]:
        """
        Get comprehensive UI state from browser
        
        Returns:
            Complete UI state dictionary
        """
        if not self.page:
            return {}
        
        try:
            workflow = await self.get_current_workflow()
            sidebar = await self.get_sidebar_state()
            canvas = await self.get_canvas_info()
            agent = await self.get_agent_status()
            
            return {
                "current_workflow": workflow,
                "sidebar_open": sidebar["sidebar_open"],
                "sidebar_tab": sidebar["active_tab"],
                "node_count": canvas["node_count"],
                "zoom_level": canvas["zoom_level"],
                "links_visible": canvas["links_visible"],
                "agent_connected": agent["agent_connected"],
                "client_count": agent["client_count"],
                "selected_client": agent.get("selected_client", "Local")
            }
        except Exception as e:
            logger.error(f"Failed to get UI state: {e}")
            return {}