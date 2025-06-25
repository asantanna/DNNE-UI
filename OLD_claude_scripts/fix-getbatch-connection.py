#!/usr/bin/env python3
"""
Fix GetBatchNode to accept DATALOADER input and add BATCH type
"""

import re
import os
from datetime import datetime

def fix_getbatch_connection():
    """Fix the GetBatchNode input type issue"""
    print("=== Fixing GetBatchNode Connection Issue ===\n")
    
    # 1. Add BATCH type if missing
    add_batch_type()
    
    # 2. Fix GetBatchNode input
    fix_getbatch_input()
    
    print("\n=== Fix Complete ===")
    print("Please restart the DNNE server and refresh your browser")

def add_batch_type():
    """Add BATCH type to robotics types"""
    types_file = "custom_nodes/robotics_nodes/robotics_types.py"
    
    if not os.path.exists(types_file):
        print(f"ERROR: {types_file} not found")
        return
    
    with open(types_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if '"BATCH"' not in content:
        # Add BATCH type
        pattern = r'(ROBOTICS_BASE_TYPES\s*=\s*{[^}]*)(})'
        replacement = r'\1    "BATCH": "BATCH",  # Batch of data (images, labels)\n\2'
        content = re.sub(pattern, replacement, content)
        
        # Create backup
        backup_file = f"{types_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created backup: {backup_file}")
        
        with open(types_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✓ Added BATCH type to robotics_types.py")
    else:
        print("✓ BATCH type already exists")

def fix_getbatch_input():
    """Fix GetBatchNode to accept DATALOADER instead of TENSOR"""
    ml_nodes_file = "custom_nodes/ml_nodes/__init__.py"
    
    if not os.path.exists(ml_nodes_file):
        print(f"ERROR: {ml_nodes_file} not found")
        return
    
    with open(ml_nodes_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create backup
    backup_file = f"{ml_nodes_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\nCreated backup: {backup_file}")
    
    # Find GetBatchNode class and fix its INPUT_TYPES
    # Pattern to find the GetBatchNode INPUT_TYPES method
    pattern = r'(class\s+GetBatchNode.*?def\s+INPUT_TYPES.*?return\s*{\s*"required":\s*{\s*)"input":\s*\("TENSOR",\s*\)'
    
    # Check if the pattern exists
    if re.search(pattern, content, re.DOTALL):
        # Replace with correct input type
        replacement = r'\1"dataloader": ("DATALOADER",)'
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        print("✓ Fixed GetBatchNode to accept DATALOADER input")
        
        # Also fix the function parameter name
        # Find the get_batch method and fix its parameter
        func_pattern = r'(def\s+get_batch\s*\(\s*self\s*,\s*)dataloader(\s*\):)'
        if not re.search(func_pattern, content):
            # The parameter might be named differently
            func_pattern = r'(def\s+get_batch\s*\(\s*self\s*,\s*)\w+(\s*\):)'
            content = re.sub(func_pattern, r'\1dataloader\2', content)
            print("✓ Fixed get_batch method parameter name")
    else:
        print("WARNING: Could not find the expected GetBatchNode INPUT_TYPES pattern")
        print("Looking for alternative patterns...")
        
        # Try a more general pattern
        general_pattern = r'(class\s+GetBatchNode.*?)"input":\s*\("TENSOR",\s*\)'
        if re.search(general_pattern, content, re.DOTALL):
            content = re.sub(general_pattern, r'\1"dataloader": ("DATALOADER",)', content, flags=re.DOTALL)
            print("✓ Fixed GetBatchNode input type (alternative pattern)")
    
    # Save the fixed file
    with open(ml_nodes_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✓ All fixes applied!")

if __name__ == "__main__":
    fix_getbatch_connection()