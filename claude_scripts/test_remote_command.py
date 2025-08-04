#!/usr/bin/env python3
"""
Test script for the DNNE remote command endpoint.
"""

import requests
import json
import sys
import os
import time
from datetime import datetime

# Server URL
BASE_URL = "http://172.22.160.1:8188"  # Windows host from WSL2
ENDPOINT = f"{BASE_URL}/remote_command"  # Will also be available at /api/remote_command

# Optional authentication token
AUTH_TOKEN = os.environ.get("DNNE_REMOTE_AUTH")

def send_command(command, args=None, auth=None):
    """Send a command to the remote command endpoint."""
    payload = {
        "command": command,
        "args": args or {},
        "request_id": f"test_{int(time.time())}"
    }
    
    if auth:
        payload["auth"] = auth
    
    print(f"\nSending command: {command}")
    print(f"Args: {json.dumps(args, indent=2) if args else 'None'}")
    
    try:
        response = requests.post(ENDPOINT, json=payload, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            print(f"Message: {data.get('message')}")
            if data.get('data'):
                print(f"Data: {json.dumps(data['data'], indent=2)}")
        else:
            print(f"Error Response: {response.text}")
            
        return response
        
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to server at {BASE_URL}")
        print("Make sure the DNNE server is running on Windows")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def test_get_status():
    """Test the get_status command."""
    print("\n" + "="*50)
    print("Testing get_status command")
    print("="*50)
    
    response = send_command("get_status", auth=AUTH_TOKEN)
    
    if response and response.status_code == 200:
        data = response.json()
        status_data = data.get('data', {})
        
        print("\nServer Status:")
        print(f"  Uptime: {status_data.get('uptime', 0):.1f} seconds")
        print(f"  Version: {status_data.get('version', 'unknown')}")
        print(f"  Agent Connected: {status_data.get('agent_connected', False)}")
        print(f"  Agent Status: {status_data.get('agent_status', 'unknown')}")
        print(f"  Queue Size: {status_data.get('queue_size', 0)}")
        print(f"  Node Count: {status_data.get('node_count', 0)}")

def test_reload_nodes():
    """Test the reload_nodes command."""
    print("\n" + "="*50)
    print("Testing reload_nodes command")
    print("="*50)
    
    response = send_command("reload_nodes", auth=AUTH_TOKEN)
    
    if response and response.status_code == 200:
        data = response.json()
        reload_data = data.get('data', {})
        print(f"\nNodes reloaded: {reload_data.get('node_count', 0)} nodes")

def test_clear_cache():
    """Test the clear_cache command."""
    print("\n" + "="*50)
    print("Testing clear_cache command")
    print("="*50)
    
    response = send_command("clear_cache", {"type": "all"}, auth=AUTH_TOKEN)
    
    if response and response.status_code == 200:
        data = response.json()
        print(f"\nCache cleared successfully")

def test_restart(delay=3):
    """Test the restart command."""
    print("\n" + "="*50)
    print("Testing restart command")
    print("="*50)
    print(f"WARNING: This will restart the server in {delay} seconds!")
    
    confirmation = input("Are you sure you want to restart? (y/n): ")
    if confirmation.lower() != 'y':
        print("Restart cancelled")
        return
    
    response = send_command(
        "restart", 
        {
            "delay": delay,
            "reason": "Test restart from remote command",
            "preserve_args": True
        },
        auth=AUTH_TOKEN
    )
    
    if response and response.status_code == 200:
        print(f"\nServer will restart in {delay} seconds...")
        print("Waiting for server to come back online...")
        
        # Wait for restart
        time.sleep(delay + 5)
        
        # Try to connect again
        print("\nChecking if server is back online...")
        for i in range(10):
            try:
                test_response = requests.get(f"{BASE_URL}/system_stats", timeout=2)
                if test_response.status_code == 200:
                    print("Server is back online!")
                    break
            except:
                pass
            time.sleep(2)
            print(f"Waiting... ({i+1}/10)")

def test_unknown_command():
    """Test an unknown command."""
    print("\n" + "="*50)
    print("Testing unknown command (should fail)")
    print("="*50)
    
    response = send_command("unknown_command", auth=AUTH_TOKEN)
    
    if response:
        print(f"\nExpected error response received (status {response.status_code})")

def main():
    """Run all tests."""
    print("DNNE Remote Command Endpoint Test")
    print("==================================")
    print(f"Server: {BASE_URL}")
    print(f"Endpoint: {ENDPOINT}")
    print(f"Auth: {'Configured' if AUTH_TOKEN else 'Not configured'}")
    
    # Test commands
    test_get_status()
    test_reload_nodes()
    test_clear_cache()
    test_unknown_command()
    
    # Check if we should test restart (only in interactive mode)
    if sys.stdin.isatty():
        # Ask about restart test
        print("\n" + "="*50)
        test_restart_input = input("\nDo you want to test the restart command? (y/n): ")
        if test_restart_input.lower() == 'y':
            test_restart(3)
    else:
        print("\n" + "="*50)
        print("Skipping restart test (non-interactive mode)")
    
    print("\n" + "="*50)
    print("All tests completed!")

if __name__ == "__main__":
    main()