#!/usr/bin/env python3
"""
Fix indentation issues in OUTPUT_DICT
"""

import re
from pathlib import Path

def fix_file(filepath: Path) -> bool:
    """Fix indentation in a single file"""
    content = filepath.read_text()
    
    # Fix the OUTPUT_DICT indentation pattern
    # Look for OUTPUT_DICT that's indented wrong (after a blank line)
    pattern = r'\n\n\s+OUTPUT_DICT = \{\n\s+0:'
    replacement = r'\n\n    OUTPUT_DICT = {\n        0:'
    
    new_content = re.sub(pattern, replacement, content)
    
    # Also fix closing brace indentation
    pattern2 = r'OUTPUT_DICT = \{\n\s+(.+?)\n\s*\}'
    def fix_dict(match):
        lines = match.group(1).split('\n')
        fixed_lines = []
        for line in lines:
            # Ensure proper indentation for dict items
            line = line.strip()
            if line:
                fixed_lines.append('        ' + line)
        return 'OUTPUT_DICT = {\n' + '\n'.join(fixed_lines) + '\n    }'
    
    new_content = re.sub(pattern2, fix_dict, new_content, flags=re.DOTALL)
    
    if new_content != content:
        filepath.write_text(new_content)
        print(f"  Fixed {filepath.name}")
        return True
    return False

def main():
    """Main script"""
    visnode_dir = Path("/home/asantanna/DNNE/DNNE-UI/custom_nodes")
    visnode_files = list(visnode_dir.glob("*_visnode.py"))
    
    print(f"Checking {len(visnode_files)} visnode files for indentation issues...")
    
    fixed_count = 0
    for filepath in sorted(visnode_files):
        if fix_file(filepath):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")

if __name__ == "__main__":
    main()