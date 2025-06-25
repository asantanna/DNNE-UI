#!/usr/bin/env python3
"""
Robust script to remove context from DNNE nodes and templates
This version properly handles Python tuple syntax and validates changes
"""

import os
import re
import ast
import shutil
from pathlib import Path
from datetime import datetime

class ContextMigration:
    def __init__(self, dry_run=False, backup=True):
        self.dry_run = dry_run
        self.backup = backup
        self.backup_dir = None
        self.changes_made = []
        
    def create_backup(self):
        """Create backup of all files that will be modified"""
        if not self.backup:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = Path(f"backups/context_migration_{timestamp}")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating backup in {self.backup_dir}")
        
        # List of files to backup
        files_to_backup = [
            "custom_nodes/ml_nodes/__init__.py",
            "export_system/node_exporters/ml_nodes.py",
            "custom_nodes/robotics_nodes/robotics_types.py",
        ]
        
        # Add template files
        template_dir = Path("export_system/templates/nodes")
        if template_dir.exists():
            files_to_backup.extend([str(f) for f in template_dir.glob("*.py")])
        
        for filepath in files_to_backup:
            if os.path.exists(filepath):
                dest = self.backup_dir / filepath
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(filepath, dest)
                print(f"  Backed up {filepath}")
    
    def validate_tuple_syntax(self, content):
        """Check if Python code has valid tuple syntax"""
        try:
            ast.parse(content)
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    def fix_single_element_tuples(self, content):
        """Fix single-element tuples that are missing trailing commas"""
        # Pattern to find assignments that look like single-element tuples
        patterns = [
            # RETURN_TYPES = ("SOMETHING") -> RETURN_TYPES = ("SOMETHING",)
            (r'(RETURN_TYPES\s*=\s*\()(["\'][^"\']+["\'])\)', r'\1\2,)'),
            # RETURN_NAMES = ("something") -> RETURN_NAMES = ("something",)
            (r'(RETURN_NAMES\s*=\s*\()(["\'][^"\']+["\'])\)', r'\1\2,)'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        return content
    
    def remove_context_from_ml_nodes(self, filepath):
        """Remove context from ML node definitions with proper validation"""
        if not os.path.exists(filepath):
            print(f"  Skipping {filepath} (not found)")
            return False
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Step 1: Remove optional context from INPUT_TYPES
        content = re.sub(
            r'"optional":\s*{\s*"context":\s*\("CONTEXT",\s*\),?\s*}(?:,)?',
            '',
            content,
            flags=re.MULTILINE | re.DOTALL
        )
        
        # Step 2: Remove context from RETURN_TYPES
        # Handle two-element tuples becoming single-element
        content = re.sub(
            r'RETURN_TYPES\s*=\s*\((["\'][^"\']+["\'])\s*,\s*["\'"]CONTEXT["\']\s*\)',
            r'RETURN_TYPES = (\1,)',  # Note the trailing comma!
            content
        )
        
        # Handle three or more element tuples
        content = re.sub(
            r'(RETURN_TYPES\s*=\s*\([^)]*?),\s*["\'"]CONTEXT["\']\s*,',
            r'\1,',
            content
        )
        
        # Handle context at the end of a multi-element tuple
        content = re.sub(
            r'(RETURN_TYPES\s*=\s*\([^)]+),\s*["\'"]CONTEXT["\']\s*\)',
            r'\1)',
            content
        )
        
        # Step 3: Remove context from RETURN_NAMES (same pattern)
        content = re.sub(
            r'RETURN_NAMES\s*=\s*\((["\'][^"\']+["\'])\s*,\s*["\'"]context["\']\s*\)',
            r'RETURN_NAMES = (\1,)',
            content
        )
        
        content = re.sub(
            r'(RETURN_NAMES\s*=\s*\([^)]*?),\s*["\'"]context["\']\s*,',
            r'\1,',
            content
        )
        
        content = re.sub(
            r'(RETURN_NAMES\s*=\s*\([^)]+),\s*["\'"]context["\']\s*\)',
            r'\1)',
            content
        )
        
        # Step 4: Remove context parameter from function signatures
        content = re.sub(
            r'(def\s+\w+\s*\([^)]*),\s*context\s*=\s*None\s*\)',
            r'\1)',
            content
        )
        
        # Step 5: Remove context from return statements
        # Handle two-element returns becoming single
        content = re.sub(
            r'return\s*\(([^,\)]+)\s*,\s*context\s*\)',
            r'return (\1,)',
            content
        )
        
        # Handle multi-element returns
        content = re.sub(
            r'(return\s*\([^)]*?),\s*context\s*\)',
            r'\1)',
            content
        )
        
        # Step 6: Ensure no broken single-element tuples
        content = self.fix_single_element_tuples(content)
        
        # Validate the result
        valid, error = self.validate_tuple_syntax(content)
        if not valid:
            print(f"  ERROR: Invalid syntax after transformation: {error}")
            print(f"  Skipping changes to {filepath}")
            return False
        
        if content != original_content:
            if not self.dry_run:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            print(f"  {'Would update' if self.dry_run else 'Updated'} {filepath}")
            self.changes_made.append(filepath)
            return True
        else:
            print(f"  No changes needed in {filepath}")
            return False
    
    def remove_context_from_templates(self, template_dir):
        """Update template files to remove context references"""
        template_path = Path(template_dir)
        if not template_path.exists():
            print(f"  Template directory {template_dir} not found")
            return
        
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
        
        for template_file in template_path.glob("*.py"):
            self.apply_patterns_to_file(template_file, patterns)
    
    def apply_patterns_to_file(self, filepath, patterns):
        """Apply regex patterns to a file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
        
        if content != original_content:
            if not self.dry_run:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            print(f"  {'Would update' if self.dry_run else 'Updated'} {filepath}")
            self.changes_made.append(filepath)
            return True
        else:
            print(f"  No changes needed in {filepath}")
            return False
    
    def add_context_stub(self):
        """Add a stub CONTEXT type for backwards compatibility"""
        stub_code = '''
# Backwards compatibility stub for CONTEXT
class ContextStub:
    """Stub node for backwards compatibility with saved workflows containing context connections"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"context": ("CONTEXT",)}}
    
    RETURN_TYPES = ("CONTEXT",)
    RETURN_NAMES = ("context",)
    FUNCTION = "passthrough"
    CATEGORY = "hidden"
    
    def passthrough(self, context=None):
        return (context,)

# Add to NODE_CLASS_MAPPINGS if it exists
if 'NODE_CLASS_MAPPINGS' in globals():
    NODE_CLASS_MAPPINGS["ContextStub"] = ContextStub
'''
        
        stub_file = "custom_nodes/ml_nodes/context_stub.py"
        if not self.dry_run:
            os.makedirs(os.path.dirname(stub_file), exist_ok=True)
            with open(stub_file, 'w') as f:
                f.write(stub_code)
        print(f"  {'Would create' if self.dry_run else 'Created'} context stub at {stub_file}")
    
    def create_summary(self):
        """Create a summary of changes"""
        summary = f"""# Context Removal Migration Summary
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Backup Location
{'No backup created (backup=False)' if not self.backup else f'{self.backup_dir}'}

## Files Modified
Total files changed: {len(self.changes_made)}

"""
        for filepath in self.changes_made:
            summary += f"- {filepath}\n"
        
        summary += """
## What Changed
1. Removed "context" from optional inputs in node definitions
2. Removed "CONTEXT" from return types
3. Updated function signatures to remove context parameter
4. Fixed single-element tuple syntax (added trailing commas)
5. Updated templates to remove context variable handling
6. Created context stub for backwards compatibility

## Next Steps
1. Restart the DNNE server
2. Clear browser cache/localStorage if needed
3. Test that nodes load correctly
4. Test export functionality

## Rollback
To rollback changes, restore files from the backup directory.
"""
        
        if not self.dry_run:
            with open("MIGRATION_SUMMARY.md", "w") as f:
                f.write(summary)
        
        print("\n" + "="*50)
        print(summary)
    
    def run(self):
        """Run the complete migration"""
        print("Starting Context Removal Migration...")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        print("="*50 + "\n")
        
        # Check we're in the right directory
        if not os.path.exists("export_system") or not os.path.exists("custom_nodes"):
            print("ERROR: Please run this script from the DNNE-UI directory")
            return False
        
        # Create backup
        if not self.dry_run:
            self.create_backup()
        
        # Run migrations
        print("\n1. Updating ML node definitions...")
        self.remove_context_from_ml_nodes("custom_nodes/ml_nodes/__init__.py")
        
        print("\n2. Updating template files...")
        self.remove_context_from_templates("export_system/templates/nodes")
        
        print("\n3. Updating exporters...")
        patterns = [
            (r',?\s*"CONTEXT_VAR":\s*"context"', ''),
            (r'#.*context.*from connections.*\n.*\n.*\n', ''),
        ]
        self.apply_patterns_to_file("export_system/node_exporters/ml_nodes.py", patterns)
        
        print("\n4. Adding context stub for compatibility...")
        self.add_context_stub()
        
        # Create summary
        self.create_summary()
        
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove context from DNNE nodes")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backups")
    
    args = parser.parse_args()
    
    migration = ContextMigration(dry_run=args.dry_run, backup=not args.no_backup)
    success = migration.run()
    
    if success and args.dry_run:
        print("\nThis was a DRY RUN. No files were modified.")
        print("Run without --dry-run to apply changes.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())