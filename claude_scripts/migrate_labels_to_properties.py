#!/usr/bin/env python3
"""
Script to migrate DNNE workflows from dictionary-based labels to property-based labels.
This updates Label nodes to store connection info in their properties instead of
relying on a separate labelDictionary.

Usage:
    python migrate_labels_to_properties.py workflow.json
    python migrate_labels_to_properties.py workflow         # .json extension optional
    python migrate_labels_to_properties.py *.json          # Migrate multiple files
"""

import json
import sys
import os
from pathlib import Path
import argparse
from typing import Dict, Any, List


def migrate_labels_in_workflow(workflow_data: Dict[str, Any]) -> bool:
    """
    Migrate label nodes to use properties for connection info.
    Returns True if changes were made.
    
    FAIL-FAST: This function ONLY migrates from labelDictionary.
    It will NOT attempt to guess or infer connections.
    """
    changes_made = False
    
    # Get label dictionary - REQUIRED for migration
    label_dict = workflow_data.get('extra', {}).get('labelDictionary', {})
    if not label_dict:
        # No dictionary means we can't migrate
        return False
    
    # Process all Label nodes
    nodes = workflow_data.get('nodes', [])
    
    for node in nodes:
        if node.get('type') == 'Label':
            if 'properties' not in node:
                node['properties'] = {}
            
            props = node['properties']
            label_name = props.get('labelName')
            direction = props.get('labelDirection')
            
            # Skip if already migrated (has resolution properties)
            if direction == 'output' and 'sourceNodeId' in props:
                continue
            if direction == 'input' and 'targetNodeId' in props:
                continue
            
            # Find corresponding entry in label dictionary
            if label_name and label_dict:
                # For output labels, the dictionary key is the label name
                if direction == 'output' and label_name in label_dict:
                    dict_entry = label_dict[label_name]
                    props['sourceNodeId'] = dict_entry['nodeId']
                    props['sourceSlotName'] = dict_entry['slotName']
                    props['sourceSlotType'] = dict_entry.get('slotType', '*')
                    
                    # Find slot index by looking at actual connections
                    # This requires looking at links to find the connection TO this label
                    links = workflow_data.get('links', [])
                    for link in links:
                        if len(link) >= 5 and link[3] == node['id']:
                            # This link connects TO this label node
                            props['sourceSlotIndex'] = link[2]  # from_slot
                            break
                    
                    print(f"    Migrated output label '{label_name}': node {props['sourceNodeId']}")
                    changes_made = True
                
                # For input labels, need to find the dictionary entry with this node
                elif direction == 'input':
                    # Search for dictionary entries that reference this node
                    for key, dict_entry in label_dict.items():
                        if (dict_entry.get('direction') == 'input' and 
                            dict_entry.get('nodeId') == node['id']):
                            
                            props['targetNodeId'] = dict_entry['nodeId']
                            props['targetSlotName'] = dict_entry['slotName']
                            props['targetSlotType'] = dict_entry.get('slotType', '*')
                            props['connectedToLabel'] = dict_entry.get('connectedToLabel', label_name)
                            
                            # Find slot index by looking at actual connections
                            # This requires looking at links to find the connection FROM this label
                            links = workflow_data.get('links', [])
                            for link in links:
                                if len(link) >= 5 and link[1] == node['id']:
                                    # This link connects FROM this label node
                                    props['targetSlotIndex'] = link[4]  # to_slot
                                    break
                            
                            print(f"    Migrated input label connecting to '{props['connectedToLabel']}': node {props['targetNodeId']}")
                            changes_made = True
                            break
    
    # Optionally remove the labelDictionary after migration
    # (keeping it for now for backward compatibility)
    # if changes_made and 'extra' in workflow_data and 'labelDictionary' in workflow_data['extra']:
    #     del workflow_data['extra']['labelDictionary']
    #     print("  Removed labelDictionary after migration")
    
    return changes_made


def process_workflow_file(filepath: Path, backup: bool = True) -> None:
    """Process a single workflow file."""
    print(f"\nProcessing: {filepath}")
    
    try:
        # Read the workflow file
        with open(filepath, 'r') as f:
            workflow_data = json.load(f)
        
        modified = False
        
        # Create backup if requested
        if backup:
            backup_path = filepath.with_suffix('.backup.json')
            with open(backup_path, 'w') as f:
                json.dump(workflow_data, f, indent=2)
            print(f"  Created backup: {backup_path}")
        
        # Check if labels exist
        label_nodes = [n for n in workflow_data.get('nodes', []) if n.get('type') == 'Label']
        
        if not label_nodes:
            print("  No Label nodes found in workflow")
            return
        
        # Check if migration is needed
        extra = workflow_data.get("extra", {})
        label_dict = extra.get("labelDictionary", {})
        
        # Check if labels already have properties
        labels_with_props = [n for n in label_nodes if n.get("properties") and 
                            (n["properties"].get("sourceNodeId") is not None or 
                             n["properties"].get("targetNodeId") is not None)]
        
        if label_dict and labels_with_props:
            # Already migrated but dictionary still exists - remove it
            print(f"  Labels already migrated ({len(labels_with_props)} labels with properties)")
            print(f"  Removing obsolete labelDictionary...")
            del extra["labelDictionary"]
            modified = True
        elif label_dict and not labels_with_props:
            # Need to migrate
            print(f"  Found labelDictionary - migrating {len(label_nodes)} labels...")
            if migrate_labels_in_workflow(workflow_data):
                # Also remove the dictionary after migration
                if "labelDictionary" in extra:
                    del extra["labelDictionary"]
                    print("  Removed labelDictionary after migration")
                modified = True
            else:
                print("  ✗ ERROR: Migration failed")
        elif not label_dict and labels_with_props:
            print(f"  File is fully migrated already, no action taken")
        elif not label_dict and not labels_with_props:
            # FAIL FAST - can't migrate without dictionary
            print(f"  ✗ ERROR: Workflow has {len(label_nodes)} Label nodes but no labelDictionary")
            print(f"    Cannot migrate - labelDictionary is required for migration")
            return
        
        # Save if modified
        if modified:
            with open(filepath, 'w') as f:
                json.dump(workflow_data, f, indent=2)
            print(f"  ✓ Workflow saved")
            
    except json.JSONDecodeError as e:
        print(f"  ✗ Error: Invalid JSON in file - {e}")
    except Exception as e:
        print(f"  ✗ Error processing file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Migrate DNNE workflows from dictionary-based labels to property-based labels'
    )
    parser.add_argument(
        'files', 
        nargs='+', 
        help='Workflow JSON files to migrate'
    )
    parser.add_argument(
        '--no-backup', 
        action='store_true',
        help='Do not create backup files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be migrated without making changes'
    )
    
    args = parser.parse_args()
    
    # Expand wildcards and get all files
    all_files = []
    for pattern in args.files:
        if '*' in pattern:
            # Handle wildcards
            from glob import glob
            all_files.extend(glob(pattern))
        else:
            all_files.append(pattern)
    
    # Remove duplicates and convert to Path objects, handling missing .json extension
    workflow_files = []
    for f in set(all_files):
        filepath = Path(f)
        
        # If file doesn't exist and doesn't have .json extension, try adding it
        if not filepath.exists() and filepath.suffix != '.json':
            json_path = filepath.with_suffix('.json')
            if json_path.exists():
                filepath = json_path
        
        # Add to list if it exists and is a JSON file
        if filepath.exists():
            if filepath.suffix != '.json':
                print(f"Warning: {filepath} is not a .json file, skipping")
                continue
            workflow_files.append(filepath)
    
    if not workflow_files:
        print("No valid JSON workflow files found")
        return 1
    
    print(f"Found {len(workflow_files)} workflow file(s) to process")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***")
        for filepath in workflow_files:
            print(f"\nChecking: {filepath}")
            try:
                with open(filepath, 'r') as f:
                    workflow_data = json.load(f)
                migrate_labels_in_workflow(workflow_data)
            except Exception as e:
                print(f"  ✗ Error: {e}")
    else:
        for filepath in workflow_files:
            process_workflow_file(filepath, backup=not args.no_backup)
    
    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())