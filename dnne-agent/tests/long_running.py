#!/usr/bin/env python3
"""
Long-running test workflow for testing stop functionality.
Runs indefinitely until stopped.
"""

import time
import signal
import sys

# Flag for graceful shutdown
running = True

def signal_handler(signum, frame):
    global running
    print(f"Received signal {signum}, shutting down gracefully...")
    running = False

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

print("Long-running test workflow started")
print("Press Ctrl+C or send SIGTERM to stop")

counter = 0
while running:
    print(f"Still running... iteration {counter}")
    counter += 1
    
    try:
        time.sleep(2)
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
        break

print(f"Long-running test completed after {counter} iterations")
sys.exit(0)