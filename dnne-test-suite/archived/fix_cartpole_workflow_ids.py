#!/usr/bin/env python3
"""
Fix Cartpole workflow IDs to be valid UUIDs
"""

import json
import uuid
from pathlib import Path

def generate_uuid():
    """Generate a new UUID string"""
    return str(uuid.uuid4())

def fix_workflow_ids(workflow_path):
    """Fix all IDs in workflow to be valid UUIDs"""
    
    # Load workflow
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    # Generate new UUID for top-level workflow id
    workflow["id"] = generate_uuid()
    
    # Create mapping from old node IDs to new UUIDs
    old_to_new_ids = {}
    
    # First pass: generate new UUIDs for all nodes
    for node in workflow.get("nodes", []):
        old_id = node["id"]
        new_id = generate_uuid()
        # Store mapping for both string and int representations
        old_to_new_ids[old_id] = new_id
        old_to_new_ids[str(old_id)] = new_id
        if isinstance(old_id, int):
            old_to_new_ids[old_id] = new_id
        node["id"] = new_id
    
    # Second pass: update all link references
    for link in workflow.get("links", []):
        # Links format: [from_node_id, from_slot, to_node_id, to_slot, unused, type]
        if len(link) >= 4:
            from_node_id = link[0]
            to_node_id = link[2]
            
            # Update node IDs in links (handle both int and string IDs)
            if str(from_node_id) in old_to_new_ids:
                link[0] = old_to_new_ids[str(from_node_id)]
            elif from_node_id in old_to_new_ids:
                link[0] = old_to_new_ids[from_node_id]
                
            if str(to_node_id) in old_to_new_ids:
                link[2] = old_to_new_ids[str(to_node_id)]
            elif to_node_id in old_to_new_ids:
                link[2] = old_to_new_ids[to_node_id]
    
    # Third pass: update input links in nodes
    for node in workflow.get("nodes", []):
        for input_def in node.get("inputs", []):
            link_id = input_def.get("link")
            if link_id is not None:
                # Find the corresponding link and update if needed
                for i, link in enumerate(workflow.get("links", [])):
                    # If this is the link, it should already be updated above
                    pass
    
    # Fourth pass: update output links in nodes  
    for node in workflow.get("nodes", []):
        for output_def in node.get("outputs", []):
            links = output_def.get("links", [])
            # Output links just reference link indices, not node IDs, so they should be fine
    
    # Fifth pass: update extra data (reroutes, linkExtensions)
    extra = workflow.get("extra", {})
    
    # Update reroutes
    if "reroutes" in extra:
        for reroute in extra["reroutes"]:
            if "id" in reroute:
                # Generate new UUID for reroute
                reroute["id"] = generate_uuid()
            if "parentId" in reroute:
                # Update parent ID if it's a node ID
                parent_id = reroute["parentId"]
                if parent_id in old_to_new_ids:
                    reroute["parentId"] = old_to_new_ids[parent_id]
    
    # Update linkExtensions  
    if "linkExtensions" in extra:
        for extension in extra["linkExtensions"]:
            if "id" in extension:
                extension["id"] = generate_uuid()
            if "parentId" in extension:
                parent_id = extension["parentId"]
                if parent_id in old_to_new_ids:
                    extension["parentId"] = old_to_new_ids[parent_id]
    
    return workflow

def main():
    """Main function"""
    workflow_path = Path("user/default/workflows/Cartpole_RL_Single.json")
    
    print(f"Fixing workflow IDs in: {workflow_path}")
    
    # Fix the workflow
    fixed_workflow = fix_workflow_ids(workflow_path)
    
    # Create backup
    backup_path = workflow_path.with_suffix('.json.backup')
    print(f"Creating backup: {backup_path}")
    
    with open(workflow_path, 'r') as f:
        original_content = f.read()
    
    with open(backup_path, 'w') as f:
        f.write(original_content)
    
    # Write fixed workflow
    print("Writing fixed workflow...")
    with open(workflow_path, 'w') as f:
        json.dump(fixed_workflow, f, indent=2)
    
    print("✓ Workflow IDs fixed successfully!")
    print(f"✓ Backup saved to: {backup_path}")
    
    # Validate the result
    with open(workflow_path, 'r') as f:
        result = json.load(f)
    
    print(f"✓ New workflow ID: {result['id']}")
    print(f"✓ Fixed {len(result.get('nodes', []))} node IDs")
    print(f"✓ Fixed {len(result.get('links', []))} links")

if __name__ == "__main__":
    main()