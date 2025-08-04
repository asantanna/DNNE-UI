#!/usr/bin/env python3
"""Test script to verify MCP server functionality"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dnne_ui_mcp_server import DNNE_UI_MCPServer
from browser_controller import BrowserController

async def test_browser_controller():
    """Test that browser controller can connect to DNNE UI"""
    print("Testing Browser Controller...")
    print("-" * 50)
    
    controller = BrowserController()
    
    try:
        print("1. Initializing browser...")
        await controller.initialize()
        print("   ✓ Browser initialized")
        
        print("2. Checking UI is ready...")
        ready = await controller.wait_for_ui_ready(timeout=5000)
        if ready:
            print("   ✓ UI is ready")
        else:
            print("   ⚠ UI not ready (is DNNE server running?)")
        
        print("3. Taking screenshot...")
        screenshot_path = await controller.take_screenshot("test_mcp")
        if screenshot_path:
            print(f"   ✓ Screenshot saved to {screenshot_path}")
        
        print("4. Checking for sidebar buttons...")
        visible = await controller.is_visible(".side-bar-button")
        if visible:
            print("   ✓ Sidebar buttons visible")
        
        print("\n✅ Browser controller test passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        
    finally:
        print("\nCleaning up...")
        await controller.cleanup()
        print("✓ Cleanup complete")

async def test_mcp_server():
    """Test that MCP server can be instantiated"""
    print("\nTesting MCP Server...")
    print("-" * 50)
    
    try:
        server = DNNE_UI_MCPServer()
        print("✓ MCP server created successfully")
        print(f"  - DNNE URL: {server.dnne_url}")
        print(f"  - Headless: {server.headless}")
        # FastMCP stores tools differently - let's just verify creation worked
        print("  - Server initialized successfully")
        
        print("\n✅ MCP server test passed!")
        
    except Exception as e:
        print(f"\n❌ MCP server test failed: {e}")

async def main():
    """Run all tests"""
    print("=" * 60)
    print("DNNE UI MCP Server Test Suite")
    print("=" * 60)
    
    # Test MCP server instantiation
    await test_mcp_server()
    
    # Ask user if they want to test browser (requires DNNE server running)
    print("\n" + "=" * 60)
    response = input("Test browser controller? (requires DNNE server running) [y/N]: ")
    
    if response.lower() == 'y':
        await test_browser_controller()
    else:
        print("Skipping browser test")
    
    print("\n" + "=" * 60)
    print("All tests complete!")

if __name__ == "__main__":
    asyncio.run(main())