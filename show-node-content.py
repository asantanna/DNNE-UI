#!/usr/bin/env python3
"""
Show the actual content of node definitions
"""

import re
import os

def show_node_content():
    """Display the actual content of node classes"""
    print("=== Actual Node Content ===\n")
    
    ml_nodes_file = "custom_nodes/ml_nodes/__init__.py"
    
    if not os.path.exists(ml_nodes_file):
        print(f"ERROR: {ml_nodes_file} not found")
        return
    
    with open(ml_nodes_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Show file size to ensure we're reading it
    print(f"File size: {len(content)} characters\n")
    
    # Look for specific nodes and show their complete content
    nodes_to_show = ["BatchSamplerNode", "GetBatchNode", "MNISTDatasetNode"]
    
    for node_name in nodes_to_show:
        print(f"\n{'='*50}")
        print(f"Looking for {node_name}...")
        print('='*50)
        
        # Try different patterns to find the node
        patterns = [
            # Standard class definition
            rf'class\s+{node_name}[^:]*:.*?(?=\nclass|\n\n\n|\Z)',
            # With decorators
            rf'(?:@\w+\s*\n)*class\s+{node_name}[^:]*:.*?(?=\nclass|\n\n\n|\Z)',
            # Just the class line
            rf'class\s+{node_name}.*',
        ]
        
        found = False
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                found = True
                node_content = match.group(0)
                # Limit output to first 1000 chars to avoid too much
                if len(node_content) > 1000:
                    print(node_content[:1000] + "\n... (truncated)")
                else:
                    print(node_content)
                break
        
        if not found:
            print(f"Could not find {node_name}")
    
    # Also check if nodes might be imported from elsewhere
    print("\n" + "="*50)
    print("Checking for node imports...")
    print("="*50)
    
    # Look for import statements that might import nodes
    import_patterns = [
        r'from\s+\.?\w+\s+import.*Node',
        r'from\s+\S+\s+import\s+\*',
    ]
    
    for pattern in import_patterns:
        imports = re.findall(pattern, content)
        for imp in imports:
            print(f"  {imp}")
    
    # Show NODE_CLASS_MAPPINGS definition
    print("\n" + "="*50)
    print("NODE_CLASS_MAPPINGS content:")
    print("="*50)
    
    mappings_match = re.search(
        r'NODE_CLASS_MAPPINGS\s*=\s*{([^}]+)}',
        content,
        re.DOTALL
    )
    
    if mappings_match:
        mappings_content = mappings_match.group(0)
        if len(mappings_content) > 800:
            print(mappings_content[:800] + "\n... (truncated)")
        else:
            print(mappings_content)
    
    # Check if there's a __all__ export
    print("\n" + "="*50)
    print("Checking __all__ export:")
    print("="*50)
    
    all_match = re.search(r'__all__\s*=\s*\[([^\]]+)\]', content, re.DOTALL)
    if all_match:
        print(f"__all__ = [{all_match.group(1)}]")

if __name__ == "__main__":
    show_node_content()