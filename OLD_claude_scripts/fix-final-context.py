#!/usr/bin/env python3
"""
Fix the final remaining CONTEXT references in input definitions
"""

import re
import os
from datetime import datetime

def fix_final_context():
    """Fix the last remaining context references"""
    print("=== Fixing Final Context References ===\n")
    
    ml_nodes_file = "custom_nodes/ml_nodes/__init__.py"
    
    if not os.path.exists(ml_nodes_file):
        print(f"ERROR: {ml_nodes_file} not found")
        return
    
    with open(ml_nodes_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create backup
    backup_file = f"{ml_nodes_file}.backup_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created backup: {backup_file}\n")
    
    # Show the context around the remaining CONTEXT references
    print("Looking for remaining CONTEXT references...\n")
    
    lines = content.split('\n')
    context_lines = []
    
    for i, line in enumerate(lines):
        if '"context": ("CONTEXT"' in line:
            context_lines.append(i)
            # Show context around this line
            start = max(0, i - 5)
            end = min(len(lines), i + 6)
            print(f"Found at line {i+1}:")
            print("-" * 60)
            for j in range(start, end):
                prefix = ">>> " if j == i else "    "
                print(f"{prefix}{j+1}: {lines[j]}")
            print("-" * 60 + "\n")
    
    if not context_lines:
        print("No CONTEXT references found!")
        return
    
    # Fix patterns for context in inputs
    fixes_applied = 0
    
    # Pattern 1: Remove context from required inputs
    pattern1 = r'"context":\s*\("CONTEXT",\s*\),?\s*\n'
    matches = re.findall(pattern1, content)
    if matches:
        content = re.sub(pattern1, '', content)
        fixes_applied += len(matches)
        print(f"✓ Removed {len(matches)} context inputs from required sections")
    
    # Pattern 2: Remove context from any input definition (more general)
    pattern2 = r',?\s*"context":\s*\("CONTEXT",\s*\)'
    matches = re.findall(pattern2, content)
    if matches:
        content = re.sub(pattern2, '', content)
        fixes_applied += len(matches)
        print(f"✓ Removed {len(matches)} context input definitions")
    
    # Pattern 3: Clean up any trailing commas that might be left
    pattern3 = r',(\s*})'
    content = re.sub(pattern3, r'\1', content)
    
    # Pattern 4: Remove entire lines with just context
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if '"context": ("CONTEXT"' in line:
            # Skip this line entirely
            continue
        new_lines.append(line)
    
    if len(new_lines) < len(lines):
        content = '\n'.join(new_lines)
        fixes_applied += len(lines) - len(new_lines)
        print(f"✓ Removed {len(lines) - len(new_lines)} lines containing context")
    
    # Check if all CONTEXT references are gone
    print("\nVerifying fixes...")
    remaining = len(re.findall(r'"CONTEXT"', content))
    context_type_defs = len(re.findall(r'#.*CONTEXT|ROBOTICS_BASE_TYPES.*CONTEXT', content))
    actual_remaining = remaining - context_type_defs
    
    if actual_remaining == 0:
        print("✓ All CONTEXT references successfully removed!")
    else:
        print(f"⚠ Still {actual_remaining} CONTEXT references remaining")
        
        # Show where they are
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '"CONTEXT"' in line and 'ROBOTICS_BASE_TYPES' not in line and '#' not in line:
                print(f"   Line {i+1}: {line.strip()[:80]}...")
    
    # Save the file
    with open(ml_nodes_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✓ Applied {fixes_applied} fixes")
    print("\nPlease restart the DNNE server and refresh your browser")

if __name__ == "__main__":
    fix_final_context()