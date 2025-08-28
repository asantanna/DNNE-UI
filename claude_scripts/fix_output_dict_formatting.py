#!/usr/bin/env python3
"""
Fix formatting issues in OUTPUT_DICT conversion
"""

import re
from pathlib import Path

def fix_file(filepath: Path) -> bool:
    """Fix formatting in a single file"""
    content = filepath.read_text()
    
    # Fix the pattern where OUTPUT_DICT is on same line as closing brace
    pattern = r'(\s*)\}\s*OUTPUT_DICT = \{'
    replacement = r'\1}\n\n\1OUTPUT_DICT = {'
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:
        filepath.write_text(new_content)
        print(f"  Fixed {filepath.name}")
        return True
    return False

def main():
    """Main script"""
    visnode_dir = Path("/home/asantanna/DNNE/DNNE-UI/custom_nodes")
    visnode_files = list(visnode_dir.glob("*_visnode.py"))
    
    print(f"Checking {len(visnode_files)} visnode files for formatting issues...")
    
    fixed_count = 0
    for filepath in sorted(visnode_files):
        if fix_file(filepath):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")

if __name__ == "__main__":
    main()