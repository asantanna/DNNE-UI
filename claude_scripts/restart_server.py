#!/usr/bin/env python3
"""
Manually trigger a server restart using the current (old) API.
This is a temporary workaround until the new remote_command endpoint is available.
"""

import os
import sys
import time

print("Server Restart Helper")
print("=====================")
print("Please manually restart the DNNE server on Windows:")
print("")
print("1. Close the DNNE server window (Ctrl+C or close the window)")
print("2. Run dnne.bat again to restart the server")
print("")
print("The remote command endpoint will be available after restart.")
print("")
print("Once restarted, you can use the remote command endpoint to restart programmatically:")
print("  python claude_scripts/test_remote_command.py")
print("")
input("Press Enter after you've restarted the server...")