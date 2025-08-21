#!/usr/bin/env python3
"""
Proof of concept script to test Playwright with DNNE UI.
Tests basic browser automation capabilities for potential MCP development.
"""

import sys
import time
from pathlib import Path

# Check if playwright is installed
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("ERROR: Playwright is not installed!")
    print("\nTo install Playwright, run:")
    print("  pip install playwright")
    print("  playwright install chromium")
    print("\nNote: The second command downloads the Chromium browser binary.")
    sys.exit(1)

def test_dnne_ui_with_playwright():
    """Test basic Playwright interaction with DNNE UI"""
    
    print("Starting Playwright test for DNNE UI...")
    print("-" * 50)
    
    # Create screenshots directory
    screenshots_dir = Path("playwright_screenshots")
    screenshots_dir.mkdir(exist_ok=True)
    print(f"Screenshots will be saved to: {screenshots_dir.absolute()}")
    
    try:
        # Start Playwright
        with sync_playwright() as p:
            print("\n1. Launching Chromium browser...")
            browser = p.chromium.launch(
                headless=False,  # Show the browser
                args=['--start-maximized']  # Maximize window for status bar visibility
            )
            
            # Create a new page
            page = browser.new_page(no_viewport=True)  # Use full window size
            
            # Try different URLs
            urls_to_try = [
                "http://172.22.160.1:8188",
                "http://localhost:8188",
                "http://127.0.0.1:8188",
                "http://host.docker.internal:8188"  # Sometimes needed in WSL2
            ]
            
            successful_url = None
            for url in urls_to_try:
                print(f"2. Attempting to navigate to DNNE UI at {url}...")
                try:
                    # Navigate with timeout
                    response = page.goto(url, timeout=5000, wait_until="domcontentloaded")
                    if response:
                        print(f"   Response status: {response.status}")
                        if response.status == 200:
                            print(f"   ✓ Successfully connected to {url}")
                            successful_url = url
                            break
                        else:
                            print(f"   ⚠ Server returned status {response.status}")
                    else:
                        print("   ⚠ No response received")
                except Exception as e:
                    print(f"   ✗ Failed: {e}")
            
            if not successful_url:
                print("\n   ✗ ERROR: Could not connect to DNNE server!")
                print("   Please ensure:")
                print("     1. DNNE server is running (./dnne.bat on Windows)")
                print("     2. Server is listening on port 8188")
                print("     3. No firewall is blocking the connection")
                print("\n   You can test connectivity from WSL2 with:")
                print("     curl http://172.22.160.1:8188")
            
            # Wait for page to load
            print("3. Waiting for page to load...")
            page.wait_for_timeout(2000)
            
            # Check current URL
            current_url = page.url
            print(f"   Current URL: {current_url}")
            
            if current_url == "about:blank":
                print("   ⚠ Browser is still on blank page - navigation failed")
            
            # Take initial screenshot
            print("4. Taking initial screenshot...")
            page.screenshot(path=screenshots_dir / "01_initial.png")
            print(f"   ✓ Saved: 01_initial.png")
            
            # Try to click on Workflows sidebar
            print("\n5. Attempting to click Workflows sidebar tab...")
            try:
                # Look for the workflows button
                workflows_btn = page.locator('.workflows-tab-button')
                if workflows_btn.count() > 0:
                    workflows_btn.click()
                    print("   ✓ Clicked Workflows tab")
                    
                    # Wait for sidebar animation
                    page.wait_for_timeout(1000)
                    
                    # Take screenshot with workflows sidebar open
                    page.screenshot(path=screenshots_dir / "02_workflows_open.png")
                    print(f"   ✓ Saved: 02_workflows_open.png")
                else:
                    print("   ⚠ Workflows button not found")
            except Exception as e:
                print(f"   ✗ Error clicking workflows: {e}")
            
            # Try to click on Node Library sidebar
            print("\n6. Attempting to click Node Library sidebar tab...")
            try:
                # Look for the node library button
                node_lib_btn = page.locator('.node-library-tab-button')
                if node_lib_btn.count() > 0:
                    node_lib_btn.click()
                    print("   ✓ Clicked Node Library tab")
                    
                    # Wait for sidebar animation
                    page.wait_for_timeout(1000)
                    
                    # Take screenshot with node library open
                    page.screenshot(path=screenshots_dir / "03_node_library_open.png")
                    print(f"   ✓ Saved: 03_node_library_open.png")
                else:
                    print("   ⚠ Node Library button not found")
            except Exception as e:
                print(f"   ✗ Error clicking node library: {e}")
            
            # Check if we can see the status bar
            print("\n7. Checking for status bar visibility...")
            try:
                # Try to get the page dimensions
                dimensions = page.evaluate('''() => {
                    return {
                        width: window.innerWidth,
                        height: window.innerHeight,
                        statusBar: document.querySelector('.status-bar') ? 'found' : 'not found'
                    }
                }''')
                print(f"   Window size: {dimensions['width']}x{dimensions['height']}")
                print(f"   Status bar element: {dimensions['statusBar']}")
            except Exception as e:
                print(f"   ✗ Error checking dimensions: {e}")
            
            # Try using Playwright's text locator to find status elements
            print("\n8. Looking for DNNE UI elements...")
            try:
                # Check for "Agent:" text which should be in status bar
                if page.get_by_text("Agent:").count() > 0:
                    print("   ✓ Found 'Agent:' text (status bar likely visible)")
                else:
                    print("   ⚠ Could not find 'Agent:' text")
                
                # Check for Export button
                if page.get_by_text("Export").count() > 0:
                    print("   ✓ Found 'Export' button")
                else:
                    print("   ⚠ Could not find 'Export' button")
            except Exception as e:
                print(f"   ✗ Error searching for elements: {e}")
            
            print("\n" + "=" * 50)
            print("TEST COMPLETE!")
            print("=" * 50)
            print(f"\nScreenshots saved to: {screenshots_dir.absolute()}")
            print("\nPlease check the screenshots to verify:")
            print("  1. The UI loaded correctly")
            print("  2. The status bar is visible")
            print("  3. Sidebar tabs responded to clicks")
            
            print("\nKeeping browser open for 3 seconds for observation...")
            page.wait_for_timeout(3000)
            
            # Close browser
            browser.close()
            print("Browser closed.")
            
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("DNNE UI Playwright Test")
    print("=" * 50)
    success = test_dnne_ui_with_playwright()
    sys.exit(0 if success else 1)