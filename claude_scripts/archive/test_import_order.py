#!/usr/bin/env python3
"""Test script to verify Isaac Gym import order"""

import sys
from pathlib import Path

# Test 1: Verify importing isaacgym before torch works
print("Test 1: Import isaacgym before torch")
try:
    import isaacgym
    print("✓ isaacgym imported successfully")
    import torch
    print("✓ torch imported successfully after isaacgym")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Test importing from a generated export
print("\nTest 2: Import from Cartpole_PPO_Test2 export")
export_path = Path("/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO_Test2")
if export_path.exists():
    sys.path.insert(0, str(export_path))
    try:
        # This should work if nodes are ordered correctly
        from nodes import *
        print("✓ Successfully imported all nodes from export")
        
        # Check which nodes were imported
        import nodes
        node_classes = [name for name in dir(nodes) if name.endswith('Node') and not name.startswith('_')]
        print(f"✓ Imported node classes: {', '.join(node_classes)}")
        
    except Exception as e:
        print(f"✗ Error importing nodes: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"✗ Export path not found: {export_path}")

# Test 3: Check module import order
print("\nTest 3: Check sys.modules order")
isaac_modules = [m for m in sys.modules if 'isaac' in m.lower()]
torch_modules = [m for m in sys.modules if 'torch' in m]

print(f"Isaac modules loaded: {len(isaac_modules)}")
print(f"Torch modules loaded: {len(torch_modules)}")

# Find when each was loaded
module_order = list(sys.modules.keys())
first_isaac = -1
first_torch = -1

for i, m in enumerate(module_order):
    if 'isaac' in m.lower() and first_isaac == -1:
        first_isaac = i
        print(f"First Isaac module at position {i}: {m}")
    if 'torch' in m and first_torch == -1:
        first_torch = i
        print(f"First torch module at position {i}: {m}")

if first_isaac >= 0 and first_torch >= 0:
    if first_isaac < first_torch:
        print("✓ Isaac modules loaded before torch modules")
    else:
        print("✗ Torch modules loaded before Isaac modules")