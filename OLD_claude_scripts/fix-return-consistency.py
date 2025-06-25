#!/usr/bin/env python3
"""
Fix return types and names consistency in ml_nodes
"""

import re
import os

def fix_return_consistency():
    """Fix return types and names to be consistent"""
    print("=== Fixing Return Types and Names Consistency ===\n")
    
    ml_nodes_file = "custom_nodes/ml_nodes/__init__.py"
    
    if not os.path.exists(ml_nodes_file):
        print(f"ERROR: {ml_nodes_file} not found")
        return
    
    with open(ml_nodes_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Show the problematic section
    lines = content.split('\n')
    print("Found issue around line 412:")
    for i in range(410, 420):
        if i < len(lines):
            prefix = ">>> " if i in [411, 412] else "    "
            print(f"{prefix}{i+1}: {lines[i]}")
    
    print("\nApplying fixes...")
    
    # Fix 1: Replace invalid empty tuple syntax
    content = re.sub(r'RETURN_TYPES\s*=\s*\(\s*,\s*\)', 'RETURN_TYPES = ()', content)
    print("✓ Fixed empty tuple syntax")
    
    # Fix 2: Remove RETURN_NAMES when RETURN_TYPES is empty
    # Pattern to find RETURN_TYPES = () followed by RETURN_NAMES
    pattern = r'(RETURN_TYPES\s*=\s*\(\s*\)\s*\n\s*)RETURN_NAMES\s*=\s*\([^)]*\)'
    
    def remove_return_names(match):
        return match.group(1) + '# RETURN_NAMES removed (no outputs)'
    
    matches = re.findall(pattern, content)
    if matches:
        content = re.sub(pattern, remove_return_names, content)
        print(f"✓ Removed {len(matches)} orphaned RETURN_NAMES")
    
    # Fix 3: Look for any other nodes that might have mismatched return types/names
    print("\nChecking for other inconsistencies...")
    
    # Find all RETURN_TYPES = () occurrences
    empty_returns = []
    for i, line in enumerate(lines):
        if re.search(r'RETURN_TYPES\s*=\s*\(\s*\)', line):
            empty_returns.append(i)
    
    # Check if they have RETURN_NAMES following them
    for line_num in empty_returns:
        if line_num + 1 < len(lines):
            next_line = lines[line_num + 1]
            if 'RETURN_NAMES' in next_line:
                print(f"  Found inconsistency at line {line_num + 2}")
    
    # Fix specific node issues
    # TrainingStepNode should return loss value
    pattern = r'(class\s+TrainingStepNode.*?)RETURN_TYPES\s*=\s*\(\s*\).*?RETURN_NAMES\s*=\s*\([^)]*\)'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(
            pattern, 
            r'\1RETURN_TYPES = ("FLOAT",)\n    RETURN_NAMES = ("loss",)',
            content,
            flags=re.DOTALL
        )
        print("✓ Fixed TrainingStepNode to return FLOAT (loss)")
    
    # Save the fixed file
    with open(ml_nodes_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✓ All fixes applied!")
    print("\nPlease restart the DNNE server")

if __name__ == "__main__":
    fix_return_consistency()