#!/usr/bin/env python3
"""Simple test to check if torch was imported before isaacgym"""

import sys

# Check if torch is already imported
if 'torch' in sys.modules:
    print("ERROR: torch is already imported!")
    print("Modules with 'torch' in name:")
    for m in sys.modules:
        if 'torch' in m:
            print(f"  - {m}")
else:
    print("✓ torch is not imported yet")

# Now try to import isaacgym
try:
    import isaacgym
    print("✓ isaacgym imported successfully")
except Exception as e:
    print(f"✗ Error importing isaacgym: {e}")
    if "imported before isaacgym" in str(e):
        print("\nThis confirms the import order issue!")
        
# Now check torch again
if 'torch' in sys.modules:
    print("torch is now imported (possibly by isaacgym)")
else:
    print("torch is still not imported")