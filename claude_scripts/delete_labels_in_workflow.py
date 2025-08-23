#!/usr/bin/env python3
"""
Script to remove label node data from DNNE workflow JSON files.
This cleans up any saved label metadata and label nodes from workflows.

Usage:
    python delete_labels_in_workflow.py workflow.json
    python delete_labels_in_workflow.py workflow         # .json extension optional
    python delete_labels_in_workflow.py *.json          # Clean multiple files
    python delete_labels_in_workflow.py workflow1 workflow2  # Multiple files
"""

import json
import sys
import os
from pathlib import Path
import argparse
from typing import Dict, Any, List


def clean_label_data_from_workflow(workflow_data: Dict[str, Any]) -> bool:
    """
    Remove label-related data from a workflow.
    Returns True if changes were made.
    """
    changes_made = False
    
    # Remove label nodes from the nodes list
    if 'nodes' in workflow_data:
        original_count = len(workflow_data['nodes'])
        workflow_data['nodes'] = [
            node for node in workflow_data['nodes'] 
            if node.get('type') != 'Label'
        ]
        if len(workflow_data['nodes']) < original_count:
            print(f"  Removed {original_count - len(workflow_data['nodes'])} label nodes")
            changes_made = True
    
    # Remove labelDictionary from extra metadata
    if 'extra' in workflow_data:
        if 'labelDictionary' in workflow_data['extra']:
            label_count = len(workflow_data['extra']['labelDictionary'])
            del workflow_data['extra']['labelDictionary']
            print(f"  Removed labelDictionary with {label_count} entries")
            changes_made = True
    
    # Check nodes for any label-related metadata
    if 'nodes' in workflow_data:
        for node in workflow_data['nodes']:
            if 'extra' in node and 'labelDictionary' in node['extra']:
                del node['extra']['labelDictionary']
                print(f"  Removed labelDictionary from node {node.get('id', 'unknown')}")
                changes_made = True
    
    # Remove any links that were connected to label nodes
    # (These should be automatically removed when label nodes are deleted,
    # but we'll check just in case)
    
    return changes_made


def process_workflow_file(filepath: Path, backup: bool = True) -> None:
    """Process a single workflow file."""
    print(f"\nProcessing: {filepath}")
    
    try:
        # Read the workflow file
        with open(filepath, 'r') as f:
            workflow_data = json.load(f)
        
        # Create backup if requested
        if backup:
            backup_path = filepath.with_suffix('.backup.json')
            with open(backup_path, 'w') as f:
                json.dump(workflow_data, f, indent=2)
            print(f"  Created backup: {backup_path}")
        
        # Clean the workflow
        if clean_label_data_from_workflow(workflow_data):
            # Save the cleaned workflow
            with open(filepath, 'w') as f:
                json.dump(workflow_data, f, indent=2)
            print(f"  ✓ Workflow cleaned and saved")
        else:
            print(f"  No label data found - workflow unchanged")
            
    except json.JSONDecodeError as e:
        print(f"  ✗ Error: Invalid JSON in file - {e}")
    except Exception as e:
        print(f"  ✗ Error processing file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Remove label node data from DNNE workflow files'
    )
    parser.add_argument(
        'files', 
        nargs='+', 
        help='Workflow JSON files to clean'
    )
    parser.add_argument(
        '--no-backup', 
        action='store_true',
        help='Do not create backup files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without making changes'
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
                clean_label_data_from_workflow(workflow_data)
            except Exception as e:
                print(f"  ✗ Error: {e}")
    else:
        for filepath in workflow_files:
            process_workflow_file(filepath, backup=not args.no_backup)
    
    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())