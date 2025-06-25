#!/usr/bin/env python3
"""
Script to remove context from DNNE nodes and templates
Run this from your DNNE-UI directory
"""

import os
import re
from pathlib import Path

def remove_context_from_file(filepath, patterns):
    """Remove context-related patterns from a file"""
    if not os.path.exists(filepath):
        print(f"  Skipping {filepath} (not found)")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Apply each pattern
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    
    # Only write if changed
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Updated {filepath}")
        return True
    else:
        print(f"  No changes needed in {filepath}")
        return False

def update_ml_nodes():
    """Update ML node definitions to remove context"""
    print("\n1. Updating ML node definitions...")
    
    patterns = [
        # Remove optional context from INPUT_TYPES
        (r'"optional":\s*{\s*"context":\s*\("CONTEXT",\),?\s*}(?:,)?', ''),
        
        # Remove context from RETURN_TYPES when it's the second item
        (r'RETURN_TYPES\s*=\s*\(([^,)]+),\s*"CONTEXT"\)', r'RETURN_TYPES = (\1)'),
        
        # Remove context from RETURN_NAMES when it's the second item  
        (r'RETURN_NAMES\s*=\s*\(([^,)]+),\s*"context"\)', r'RETURN_NAMES = (\1)'),
        
        # Remove context parameter from function signatures
        (r'def\s+(\w+)\(self,([^)]*),\s*context=None\):', r'def \1(self,\2):'),
        
        # Remove context from function returns
        (r'return\s*\(([^,)]+),\s*context\)', r'return (\1,)'),
    ]
    
    # Update main ml_nodes file
    ml_nodes_path = "custom_nodes/ml_nodes/__init__.py"
    remove_context_from_file(ml_nodes_path, patterns)

def update_templates():
    """Update template files to remove context references"""
    print("\n2. Updating template files...")
    
    template_dir = Path("export_system/templates/nodes")
    
    # Patterns for templates
    patterns = [
        # Remove CONTEXT_VAR from template_vars
        (r',?\s*"CONTEXT_VAR":\s*"context"', ''),
        
        # Remove CONTEXT_VAR extraction
        (r'CONTEXT_VAR\s*=\s*template_vars\["CONTEXT_VAR"\]\s*\n', ''),
        
        # Change globals()[CONTEXT_VAR] to just context
        (r'globals\(\)\[CONTEXT_VAR\]', 'context'),
        
        # Remove any remaining CONTEXT_VAR references
        (r'\bCONTEXT_VAR\b', 'context'),
    ]
    
    # Update each template
    for template_file in template_dir.glob("*.py"):
        remove_context_from_file(template_file, patterns)

def update_exporters():
    """Update exporters to not handle context connections"""
    print("\n3. Updating node exporters...")
    
    patterns = [
        # Remove context from prepare_template_vars
        (r',?\s*"CONTEXT_VAR":\s*"context"', ''),
        
        # Remove context connection handling
        (r'#.*context.*from connections.*\n.*\n.*\n', ''),
    ]
    
    ml_exporters_path = "export_system/node_exporters/ml_nodes.py"
    remove_context_from_file(ml_exporters_path, patterns)

def update_robotics_types():
    """Update robotics_types.py to mark Context as internal only"""
    print("\n4. Updating robotics types...")
    
    types_path = "custom_nodes/robotics_nodes/robotics_types.py"
    
    patterns = [
        # Update the comment for CONTEXT type
        (r'"CONTEXT":\s*"CONTEXT",\s*#.*', '"CONTEXT": "CONTEXT",  # Internal use only - not for UI connections'),
    ]
    
    remove_context_from_file(types_path, patterns)

def create_migration_summary():
    """Create a summary of changes"""
    print("\n5. Creating migration summary...")
    
    summary = """# Context Removal Migration Summary

## Changes Made:

1. **ML Nodes** (`custom_nodes/ml_nodes/__init__.py`)
   - Removed "context" from optional inputs
   - Removed "CONTEXT" from return types
   - Updated function signatures to remove context parameter
   - Updated return statements to remove context

2. **Templates** (`export_system/templates/nodes/*.py`)
   - Removed CONTEXT_VAR from template_vars
   - Changed context references to use global context
   - Cleaned up context variable extraction

3. **Exporters** (`export_system/node_exporters/ml_nodes.py`)
   - Removed context from prepare_template_vars
   - Removed context connection handling

4. **Types** (`custom_nodes/robotics_nodes/robotics_types.py`)
   - Marked CONTEXT as internal use only

## What This Means:

- Context is now implicit (global) in generated code
- UI graphs only show data flow connections
- Cleaner, more intuitive node graphs
- Context still exists in implementation but hidden from users

## Next Steps:

1. Test the updated export system
2. Update any UI code that expects context connections
3. Consider removing Context type from UI entirely
"""
    
    with open("CONTEXT_MIGRATION.md", "w") as f:
        f.write(summary)
    
    print(f"  Created CONTEXT_MIGRATION.md")

def main():
    print("Starting context removal migration...")
    print("=" * 50)
    
    # Check we're in the right directory
    if not os.path.exists("export_system") or not os.path.exists("custom_nodes"):
        print("ERROR: Please run this script from the DNNE-UI directory")
        return
    
    # Run updates
    update_ml_nodes()
    update_templates()
    update_exporters()
    update_robotics_types()
    create_migration_summary()
    
    print("\n" + "=" * 50)
    print("Migration complete!")
    print("\nIMPORTANT: Please review the changes and test the export system")
    print("Some manual adjustments may be needed for complex cases")
    print("\nSee CONTEXT_MIGRATION.md for details")

if __name__ == "__main__":
    main()