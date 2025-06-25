#!/usr/bin/env python3
"""
Diagnose DNNE UI issues - find type mismatches and context problems
"""

import re
import os
from pathlib import Path

def diagnose_issues():
    """Run diagnostics on DNNE nodes"""
    print("=== DNNE UI Diagnostics ===\n")
    
    # 1. Check node definitions
    print("1. Checking node definitions...")
    check_node_definitions()
    
    # 2. Check type definitions
    print("\n2. Checking type definitions...")
    check_type_definitions()
    
    # 3. Check for context issues
    print("\n3. Checking for context issues...")
    check_context_issues()

def check_node_definitions():
    """Check key node definitions"""
    ml_nodes_file = "custom_nodes/ml_nodes/__init__.py"
    
    if not os.path.exists(ml_nodes_file):
        print(f"  ERROR: {ml_nodes_file} not found")
        return
    
    with open(ml_nodes_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find key nodes and their types
    nodes_to_check = [
        "BatchSamplerNode",
        "GetBatchNode", 
        "MNISTDatasetNode",
        "LinearLayerNode",
        "SGDOptimizerNode",
        "TrainingStepNode"
    ]
    
    for node_name in nodes_to_check:
        print(f"\n  {node_name}:")
        
        # Find the class
        class_match = re.search(
            rf'class\s+{node_name}.*?(?=class|\Z)',
            content,
            re.DOTALL
        )
        
        if class_match:
            class_content = class_match.group(0)
            
            # Find INPUT_TYPES
            input_match = re.search(
                r'def\s+INPUT_TYPES.*?return\s*({[^}]*})',
                class_content,
                re.DOTALL
            )
            if input_match:
                inputs = input_match.group(1)
                # Extract input names and types
                input_types = re.findall(r'"(\w+)":\s*\("([^"]+)"', inputs)
                if input_types:
                    print("    Inputs:")
                    for name, type_ in input_types:
                        print(f"      - {name}: {type_}")
                else:
                    print("    Inputs: None found")
            
            # Find RETURN_TYPES
            return_match = re.search(
                r'RETURN_TYPES\s*=\s*\(([^)]*)\)',
                class_content
            )
            if return_match:
                returns = return_match.group(1)
                print(f"    Returns: {returns}")
            else:
                print("    Returns: Not found")
        else:
            print(f"    ERROR: Class not found")

def check_type_definitions():
    """Check what types are defined"""
    types_file = "custom_nodes/robotics_nodes/robotics_types.py"
    
    if not os.path.exists(types_file):
        print(f"  ERROR: {types_file} not found")
        return
    
    with open(types_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all defined types
    types = re.findall(r'"(\w+)":\s*"(\w+)"', content)
    
    print("  Defined types:")
    for type_name, type_value in sorted(types):
        print(f"    - {type_name}")
    
    # Check for key types
    key_types = ["DATALOADER", "DATASET", "BATCH", "OPTIMIZER"]
    missing = [t for t in key_types if not any(type_name == t for type_name, _ in types)]
    
    if missing:
        print(f"\n  Missing types: {', '.join(missing)}")

def check_context_issues():
    """Check for remaining context issues"""
    ml_nodes_file = "custom_nodes/ml_nodes/__init__.py"
    
    if not os.path.exists(ml_nodes_file):
        return
    
    with open(ml_nodes_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find nodes that only return CONTEXT
    context_only = re.findall(
        r'class\s+(\w+Node).*?RETURN_TYPES\s*=\s*\("CONTEXT",\)',
        content,
        re.DOTALL
    )
    
    if context_only:
        print("  Nodes that only return CONTEXT:")
        for node in context_only:
            print(f"    - {node}")
    else:
        print("  No nodes found that only return CONTEXT")
    
    # Find nodes with context in optional inputs
    context_inputs = re.findall(
        r'class\s+(\w+Node).*?"optional":\s*{[^}]*"context":\s*\("CONTEXT"',
        content,
        re.DOTALL
    )
    
    if context_inputs:
        print("\n  Nodes with optional context input:")
        for node in context_inputs:
            print(f"    - {node}")

if __name__ == "__main__":
    diagnose_issues()