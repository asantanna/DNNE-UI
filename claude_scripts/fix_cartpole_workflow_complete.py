#!/usr/bin/env python3
"""
Complete fix for Cartpole workflow - fix all IDs that need to be UUIDs
"""

import json
import uuid
from pathlib import Path

def generate_uuid():
    """Generate a new UUID string"""
    return str(uuid.uuid4())

def fix_workflow_complete(workflow_path):
    """Fix all IDs that need to be UUIDs based on schema requirements"""
    
    # Load workflow
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    print("Fixing workflow IDs comprehensively...")
    
    # 1. Fix top-level workflow ID
    current_id = workflow.get("id", "")
    try:
        uuid.UUID(current_id)
        print(f"✓ Workflow ID is already valid: {current_id}")
    except ValueError:
        workflow["id"] = generate_uuid()
        print(f"✓ Fixed workflow ID: {workflow['id']}")
    
    # 2. Fix reroutes IDs in extra section
    extra = workflow.get("extra", {})
    reroutes = extra.get("reroutes", [])
    if reroutes:
        print(f"Fixing {len(reroutes)} reroute IDs...")
        for reroute in reroutes:
            if "id" in reroute:
                old_id = reroute["id"]
                new_id = generate_uuid()
                reroute["id"] = new_id
                print(f"  Reroute {old_id} → {new_id}")
    
    # 3. Fix linkExtensions IDs in extra section
    link_extensions = extra.get("linkExtensions", [])
    if link_extensions:
        print(f"Fixing {len(link_extensions)} linkExtension IDs...")
        for ext in link_extensions:
            if "id" in ext:
                old_id = ext["id"]
                new_id = generate_uuid()
                ext["id"] = new_id
                print(f"  LinkExt {old_id} → {new_id}")
    
    # 4. Keep node IDs as integers (they should stay as they are)
    nodes = workflow.get("nodes", [])
    print(f"✓ Keeping {len(nodes)} node IDs as integers (correct format)")
    
    return workflow

def main():
    """Main function"""
    workflow_path = Path("user/default/workflows/Cartpole_RL_Single.json")
    
    print(f"Comprehensive fix for: {workflow_path}")
    print("=" * 50)
    
    # Fix the workflow
    fixed_workflow = fix_workflow_complete(workflow_path)
    
    # Create backup
    backup_path = workflow_path.with_suffix('.json.complete_backup')
    print(f"Creating backup: {backup_path}")
    
    with open(workflow_path, 'r') as f:
        original_content = f.read()
    
    with open(backup_path, 'w') as f:
        f.write(original_content)
    
    # Write fixed workflow
    print("Writing fixed workflow...")
    with open(workflow_path, 'w') as f:
        json.dump(fixed_workflow, f, indent=2)
    
    print("=" * 50)
    print("✓ Complete workflow fix successful!")
    print(f"✓ Backup saved to: {backup_path}")
    
    # Validate the result
    with open(workflow_path, 'r') as f:
        result = json.load(f)
    
    print(f"✓ Workflow ID: {result['id']}")
    
    # Check extra sections
    extra = result.get("extra", {})
    reroutes = extra.get("reroutes", [])
    link_extensions = extra.get("linkExtensions", [])
    
    print(f"✓ Reroutes: {len(reroutes)} with UUID IDs")
    print(f"✓ LinkExtensions: {len(link_extensions)} with UUID IDs")
    
    # Sample a few to verify UUIDs
    if reroutes:
        sample_reroute = reroutes[0]["id"]
        try:
            uuid.UUID(sample_reroute)
            print(f"✓ Sample reroute ID valid: {sample_reroute}")
        except:
            print(f"✗ Sample reroute ID invalid: {sample_reroute}")
    
    if link_extensions:
        sample_ext = link_extensions[0]["id"] 
        try:
            uuid.UUID(sample_ext)
            print(f"✓ Sample linkExt ID valid: {sample_ext}")
        except:
            print(f"✗ Sample linkExt ID invalid: {sample_ext}")

if __name__ == "__main__":
    main()