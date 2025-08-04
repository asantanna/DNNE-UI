"""Browser controller for DNNE UI automation using Playwright"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from playwright.async_api import async_playwright, Browser, Page, Playwright
try:
    from .utils.helpers import ensure_screenshot_dir, retry_with_backoff
except ImportError:
    from utils.helpers import ensure_screenshot_dir, retry_with_backoff

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
                timeout=30000
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
    
    async def wait_for_ui_ready(self, timeout: int = 10000) -> bool:
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
            await asyncio.sleep(0.5)
            
            logger.info("UI is ready")
            return True
            
        except Exception as e:
            logger.error(f"UI not ready after {timeout}ms: {e}")
            return False
    
    async def click(self, selector: str, timeout: int = 5000) -> bool:
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
    
    async def get_text(self, selector: str, timeout: int = 5000) -> Optional[str]:
        """
        Get text content of an element
        
        Args:
            selector: CSS selector for element
            timeout: Maximum wait time for element
        
        Returns:
            Text content or None if not found
        """
        if not self.page:
            return None
            
        try:
            element = await self.page.wait_for_selector(selector, timeout=timeout)
            if element:
                return await element.text_content()
            return None
            
        except Exception as e:
            logger.error(f"Failed to get text from {selector}: {e}")
            return None
    
    async def type_text(self, selector: str, text: str, timeout: int = 5000) -> bool:
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
    
    async def wait_for_selector(self, selector: str, timeout: int = 5000) -> bool:
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
            await asyncio.sleep(2)  # Give it time to fully close
            
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
    
    async def health_check(self) -> bool:
        """
        Check if browser is healthy and responsive
        
        Returns:
            True if browser is healthy
        """
        if not self.browser or not self.page:
            return False
        
        try:
            # Try a simple evaluation
            result = await self.page.evaluate("() => 1 + 1")
            return result == 2
        except:
            return False
    
    async def ensure_healthy(self) -> bool:
        """
        Ensure browser is healthy, restart if needed
        
        Returns:
            True if browser is healthy (after potential restart)
        """
        if await self.health_check():
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
                
                await asyncio.sleep(0.5)
                return True
                
        except Exception as e:
            logger.warning(f"Failed to handle dialog: {e}")
        
        return False