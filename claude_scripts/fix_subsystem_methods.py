#!/usr/bin/env python3
"""Fix get_subsystem methods to be classmethods"""

import os
import re
from pathlib import Path

def fix_file(filepath):
    """Fix get_subsystem to be a classmethod"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern to find def get_subsystem without @classmethod
    pattern = r'(\n    )def get_subsystem\((?:self|cls)\):'
    replacement = r'\1@classmethod\n\1def get_subsystem(cls):'
    
    # Check if already has @classmethod
    if '@classmethod\n    def get_subsystem' in content:
        print(f"  ✓ {filepath.name} already has @classmethod")
        return False
    
    # Apply the fix
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"  ✓ Fixed {filepath.name}")
        return True
    else:
        print(f"  ⚠ No changes needed for {filepath.name}")
        return False

def main():
    """Fix all exporter files"""
    exporters_dir = Path("/home/asantanna/DNNE/DNNE-UI/export_system/node_exporters")
    
    fixed_count = 0
    for filepath in exporters_dir.glob("*_exporter.py"):
        if fix_file(filepath):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")

if __name__ == "__main__":
    main()