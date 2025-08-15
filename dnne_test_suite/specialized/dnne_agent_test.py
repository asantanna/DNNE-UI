#!/usr/bin/env python3
"""
Automated test runner for DNNE Agent system
Checks server connectivity and runs client tests
"""

import asyncio
import websockets
import json
import sys
import subprocess
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Test configuration
CLIENT_PORT = 8766
UI_PORT = 8767
CONNECTION_TIMEOUT = 5.0

def get_windows_host_ip():
    """Get Windows host IP address when running in WSL"""
    try:
        # Try to get the default gateway which is usually the Windows host in WSL
        result = subprocess.run(['ip', 'route'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'default' in line:
                    # Extract IP from line like: "default via 172.22.160.1 dev eth0"
                    parts = line.split()
                    if len(parts) >= 3 and parts[0] == 'default' and parts[1] == 'via':
                        return parts[2]
    except Exception:
        pass
    
    # Fall back to hardcoded IP if detection fails
    return "172.22.160.1"

# Get the server host IP
SERVER_HOST = get_windows_host_ip()

# Exit codes
EXIT_SUCCESS = 0
EXIT_SERVER_NOT_RUNNING = 1
EXIT_TEST_FAILED = 2


async def check_server_connectivity():
    """Check if dnne_agent_server is running and reachable"""
    print(f"🔍 Checking dnne_agent_server connectivity at {SERVER_HOST}:{CLIENT_PORT}...")
    
    try:
        # Try to connect to the server
        uri = f"ws://{SERVER_HOST}:{CLIENT_PORT}"
        async with websockets.connect(uri, timeout=CONNECTION_TIMEOUT) as websocket:
            # Send a test registration
            await websocket.send(json.dumps({
                "type": "register",
                "hostname": "test_runner",
                "capabilities": {
                    "platform": "linux",
                    "test_mode": True
                }
            }))
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=CONNECTION_TIMEOUT)
            data = json.loads(response)
            
            if data.get("type") == "registered":
                print(f"✅ Server is running and responsive (client_id: {data.get('client_id')})")
                return True
            else:
                print("❌ Server responded but registration failed")
                return False
                
    except (asyncio.TimeoutError, ConnectionRefusedError, OSError) as e:
        print(f"❌ Cannot connect to dnne_agent_server on {SERVER_HOST}:{CLIENT_PORT}")
        print(f"   Error: {type(e).__name__}: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def run_client_tests():
    """Run the dnne_agent_client tests"""
    print("\n🧪 Running dnne_agent_client tests...")
    print("=" * 60)
    
    # Path to test script
    test_script = Path(__file__).parent / "dnne_agent" / "test_dnne_agent_client.py"
    
    if not test_script.exists():
        print(f"❌ Test script not found: {test_script}")
        return False
    
    # Run the test script with specific test cases
    cmd = [
        sys.executable,
        str(test_script),
        "--server-host", SERVER_HOST,
        "--run-tests",  # Run automated tests
        "--test-basic",  # Basic connectivity test
        "--test-deploy",  # Deployment test
        "--test-workflow",  # Workflow execution test
        "--test-telemetry"  # Telemetry test
    ]
    
    try:
        # Run the tests without capture to allow real-time output
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("\n✅ All dnne_agent tests passed!")
            return True
        else:
            print(f"\n❌ Tests failed with exit code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Failed to run tests: {e}")
        return False


async def main():
    """Main test runner"""
    print("🚀 DNNE Agent System Tests")
    print("=" * 60)
    print()
    
    # Check if server is running
    server_running = await check_server_connectivity()
    
    if not server_running:
        print("\n" + "=" * 60)
        print("⚠️  DNNE Agent Test Server Not Running!")
        print("=" * 60)
        print("\nTo run these tests, you must first start the test server on Windows:")
        print("1. Open a Windows terminal")
        print("2. Navigate to your DNNE-UI directory")
        print("3. Run: python dnne_test_suite/specialized/dnne_agent/test_dnne_agent_server.py")
        print("\nThen run this test again from Linux/WSL.")
        print("=" * 60)
        return EXIT_SERVER_NOT_RUNNING
    
    # Run client tests
    tests_passed = run_client_tests()
    
    print("\n" + "=" * 60)
    if tests_passed:
        print("✅ DNNE Agent system tests completed successfully!")
        return EXIT_SUCCESS
    else:
        print("❌ DNNE Agent system tests failed!")
        return EXIT_TEST_FAILED


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)