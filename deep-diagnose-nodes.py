#!/usr/bin/env python3
"""
Deep diagnostic to understand the actual node structure
"""

import re
import os
import ast
from pathlib import Path

def deep_diagnose():
    """Perform deep analysis of node structure"""
    print("=== Deep Node Structure Analysis ===\n")
    
    ml_nodes_file = "custom_nodes/ml_nodes/__init__.py"
    
    if not os.path.exists(ml_nodes_file):
        print(f"ERROR: {ml_nodes_file} not found")
        return
    
    with open(ml_nodes_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # First, let's look for the base class
    print("1. Looking for base classes...")
    base_classes = re.findall(r'class\s+(\w*Base\w*)\s*\([^)]*\):', content)
    print(f"   Found base classes: {base_classes}")
    
    # Look for how outputs are defined
    print("\n2. Looking for output definitions...")
    
    # Check for different patterns
    patterns = [
        (r'output_types\s*=', 'output_types = '),
        (r'OUTPUT_TYPES\s*=', 'OUTPUT_TYPES = '),
        (r'def\s+OUTPUT_TYPES', 'def OUTPUT_TYPES'),
        (r'self\.output_types\s*=', 'self.output_types = '),
        (r'output\s*=\s*\[', 'output = ['),
        (r'outputs\s*=\s*\[', 'outputs = ['),
    ]
    
    for pattern, desc in patterns:
        matches = len(re.findall(pattern, content, re.IGNORECASE))
        if matches > 0:
            print(f"   Found {matches} instances of: {desc}")
    
    # Let's look at a specific node in detail
    print("\n3. Detailed analysis of BatchSamplerNode...")
    
    # Find the BatchSamplerNode class
    batch_match = re.search(
        r'class\s+BatchSamplerNode(.*?)(?=class|\Z)',
        content,
        re.DOTALL
    )
    
    if batch_match:
        batch_content = batch_match.group(0)
        print("   BatchSamplerNode structure:")
        
        # Look for any attribute assignments
        attrs = re.findall(r'^\s*(\w+)\s*=\s*(.+)$', batch_content, re.MULTILINE)
        for attr, value in attrs:
            if attr.upper() != attr:  # Skip constants
                print(f"     {attr} = {value[:50]}...")
        
        # Look for class-level attributes
        class_attrs = re.findall(r'^\s*([A-Z_]+)\s*=\s*(.+)$', batch_content, re.MULTILINE)
        for attr, value in class_attrs:
            print(f"     {attr} = {value}")
        
        # Look for methods
        methods = re.findall(r'def\s+(\w+)\s*\(', batch_content)
        print(f"     Methods: {methods}")
        
        # Check if it's using decorators or properties
        decorators = re.findall(r'@(\w+)', batch_content)
        if decorators:
            print(f"     Decorators: {decorators}")
    
    # Look for RoboticsNodeBase
    print("\n4. Looking for RoboticsNodeBase definition...")
    base_match = re.search(
        r'class\s+RoboticsNodeBase(.*?)(?=class|\Z)',
        content,
        re.DOTALL
    )
    
    if base_match:
        base_content = base_match.group(0)
        print("   RoboticsNodeBase structure:")
        
        # Look for output-related definitions
        output_patterns = [
            r'output',
            r'OUTPUT',
            r'return.*type',
            r'RETURN.*TYPE',
        ]
        
        for pattern in output_patterns:
            matches = re.findall(rf'.*{pattern}.*', base_content, re.IGNORECASE)
            for match in matches[:3]:  # Show first 3 matches
                print(f"     {match.strip()}")
    
    # Check imports
    print("\n5. Checking imports...")
    imports = re.findall(r'^(?:from|import)\s+(.+)$', content, re.MULTILINE)
    for imp in imports[:10]:  # Show first 10 imports
        print(f"   {imp}")
    
    # Look for node registration
    print("\n6. Looking for NODE_CLASS_MAPPINGS...")
    mappings = re.search(r'NODE_CLASS_MAPPINGS\s*=\s*{([^}]+)}', content, re.DOTALL)
    if mappings:
        entries = re.findall(r'"(\w+)":\s*(\w+)', mappings.group(1))
        print(f"   Found {len(entries)} node mappings")
        for name, cls in entries[:5]:  # Show first 5
            print(f"     {name} -> {cls}")

if __name__ == "__main__":
    deep_diagnose()