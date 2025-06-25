#!/usr/bin/env python3
"""
Fix remaining context inputs and outputs in DNNE nodes
"""

import re
import os
from datetime import datetime

def fix_remaining_context():
    """Fix all remaining context issues"""
    print("=== Fixing Remaining Context Issues ===\n")
    
    ml_nodes_file = "custom_nodes/ml_nodes/__init__.py"
    
    if not os.path.exists(ml_nodes_file):
        print(f"ERROR: {ml_nodes_file} not found")
        return
    
    with open(ml_nodes_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create backup
    backup_file = f"{ml_nodes_file}.backup_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created backup: {backup_file}\n")
    
    original_content = content
    
    # 1. Remove optional context inputs from all nodes
    print("1. Removing optional context inputs...")
    
    # Pattern to find optional context inputs
    pattern = r'"optional":\s*{\s*"context":\s*\("CONTEXT",\s*\),?\s*}'
    matches = len(re.findall(pattern, content))
    if matches > 0:
        content = re.sub(pattern, '', content, flags=re.MULTILINE)
        print(f"   ✓ Removed {matches} optional context inputs")
    else:
        print("   ✓ No optional context inputs found")
    
    # Clean up empty optional blocks
    pattern = r',?\s*"optional":\s*{\s*}'
    content = re.sub(pattern, '', content)
    
    # 2. Fix nodes that only return CONTEXT
    print("\n2. Fixing nodes that only return CONTEXT...")
    
    # Fix CreateContextNode
    if 'class CreateContextNode' in content:
        # Find and fix CreateContextNode
        pattern = r'(class\s+CreateContextNode.*?)RETURN_TYPES\s*=\s*\("CONTEXT",\)'
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, r'\1RETURN_TYPES = ()', content, flags=re.DOTALL)
            print("   ✓ Fixed CreateContextNode to return nothing")
        
        # Fix the return statement
        pattern = r'(class\s+CreateContextNode.*?return\s*)\([^)]*\)'
        content = re.sub(pattern, r'\1()', content, flags=re.DOTALL)
    
    # Fix SetModeNode
    if 'class SetModeNode' in content:
        # Find and fix SetModeNode
        pattern = r'(class\s+SetModeNode.*?)RETURN_TYPES\s*=\s*\("CONTEXT",\)'
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, r'\1RETURN_TYPES = ()', content, flags=re.DOTALL)
            print("   ✓ Fixed SetModeNode to return nothing")
        
        # Fix the return statement
        pattern = r'(class\s+SetModeNode.*?return\s*)\([^)]*\)'
        content = re.sub(pattern, r'\1()', content, flags=re.DOTALL)
    
    # 3. Fix any nodes that have context in their return types alongside other types
    print("\n3. Fixing nodes with CONTEXT in multi-return types...")
    
    # Pattern to find RETURN_TYPES with CONTEXT as one of multiple returns
    pattern = r'RETURN_TYPES\s*=\s*\(([^)]*)"CONTEXT"[^)]*\)'
    
    def remove_context_from_returns(match):
        types = match.group(0)
        # Extract the tuple content
        tuple_match = re.search(r'\((.*)\)', types)
        if tuple_match:
            items = tuple_match.group(1).split(',')
            # Remove CONTEXT items
            new_items = [item.strip() for item in items if 'CONTEXT' not in item]
            # Rebuild the tuple
            if len(new_items) == 0:
                return 'RETURN_TYPES = ()'
            elif len(new_items) == 1:
                return f'RETURN_TYPES = ({new_items[0]},)'  # Single item tuple needs comma
            else:
                return f'RETURN_TYPES = ({", ".join(new_items)})'
        return match.group(0)
    
    matches = len(re.findall(pattern, content))
    if matches > 0:
        content = re.sub(pattern, remove_context_from_returns, content)
        print(f"   ✓ Removed CONTEXT from {matches} multi-return types")
    else:
        print("   ✓ No multi-return CONTEXT types found")
    
    # 4. Remove context from function signatures
    print("\n4. Removing context from function signatures...")
    
    # Remove context=None parameters
    pattern = r',\s*context\s*=\s*None'
    matches = len(re.findall(pattern, content))
    if matches > 0:
        content = re.sub(pattern, '', content)
        print(f"   ✓ Removed {matches} context parameters")
    
    # 5. Fix return statements that include context
    print("\n5. Fixing return statements with context...")
    
    # Pattern to find return statements with context
    pattern = r'return\s*\(([^,)]+),\s*context\s*\)'
    matches = len(re.findall(pattern, content))
    if matches > 0:
        content = re.sub(pattern, r'return (\1,)', content)
        print(f"   ✓ Fixed {matches} return statements")
    
    # 6. Look for any remaining CONTEXT references
    print("\n6. Checking for remaining CONTEXT references...")
    
    context_count = len(re.findall(r'"CONTEXT"', content))
    if context_count > 0:
        print(f"   ⚠ Found {context_count} remaining CONTEXT references")
        # Show where they are
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '"CONTEXT"' in line and 'ROBOTICS_BASE_TYPES' not in line:
                print(f"     Line {i+1}: {line.strip()[:80]}...")
    else:
        print("   ✓ No remaining CONTEXT references found")
    
    # Save the fixed content
    if content != original_content:
        with open(ml_nodes_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("\n✓ All fixes applied!")
    else:
        print("\n✓ No changes needed")
    
    print("\nPlease restart the DNNE server and refresh your browser")

if __name__ == "__main__":
    fix_remaining_context()