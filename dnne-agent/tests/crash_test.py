#!/usr/bin/env python3
"""
Test workflow that intentionally crashes.
Used to test error handling and crash reporting.
"""

import time
import sys

print("Crash test workflow started")
print("This workflow will crash intentionally in 3 seconds...")

# Countdown
for i in range(3, 0, -1):
    print(f"Crashing in {i}...")
    time.sleep(1)

print("CRASH: Raising intentional exception")

# Raise an exception with clear error message
raise RuntimeError("This is an intentional crash for testing error handling!")

# This should never be reached
print("ERROR: This line should not execute!")
sys.exit(1)