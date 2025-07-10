#!/usr/bin/env python3
"""Test script to debug import order issue"""

import sys
from pathlib import Path

# Add export path to sys.path
export_path = Path("/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO_Test2")
sys.path.insert(0, str(export_path))

print("Testing node imports individually...")

# Test importing nodes module
try:
    import nodes
    print("✓ nodes module imported")
    
    # List what's in the nodes module
    print("\nContents of nodes module:")
    for item in dir(nodes):
        if not item.startswith('_'):
            print(f"  - {item}")
            
    # Check __all__
    if hasattr(nodes, '__all__'):
        print(f"\nnodes.__all__ = {nodes.__all__}")
        
except Exception as e:
    print(f"✗ Error importing nodes: {e}")
    import traceback
    traceback.print_exc()

# Try running the actual runner
print("\n\nTrying to run the actual runner.py...")
try:
    # Remove the export path and let runner.py handle its own imports
    sys.path.pop(0)
    
    # Change to the export directory
    import os
    os.chdir(export_path)
    
    # Run the runner in test mode
    result = os.system("python runner.py --test-mode")
    print(f"Runner exit code: {result}")
    
except Exception as e:
    print(f"✗ Error running runner: {e}")
    import traceback
    traceback.print_exc()