#!/usr/bin/env python3
"""
Collect link and slot information from all workflow JSON files.
This includes both connected and unconnected inputs/outputs.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any

def get_workflow_files(base_dir: str = "/home/asantanna/DNNE/DNNE-UI") -> List[Path]:
    """Find all workflow JSON files."""
    workflows_dir = Path(base_dir) / "user/default/workflows"
    if not workflows_dir.exists():
        print(f"Workflows directory not found: {workflows_dir}")
        return []
    
    workflow_files = list(workflows_dir.glob("*.json"))
    print(f"Found {len(workflow_files)} workflow files")
    return workflow_files

def extract_node_info(workflow_data: Dict[str, Any], workflow_name: str) -> Dict[str, Any]:
    """Extract node, link, and slot information from a workflow."""
    result = {
        "workflow": workflow_name,
        "nodes": {},
        "links": [],
        "unconnected_slots": {
            "inputs": [],
            "outputs": []
        }
    }
    
    # Build node index
    nodes_by_id = {}
    for node in workflow_data.get("nodes", []):
        node_id = node["id"]
        nodes_by_id[node_id] = {
            "type": node.get("type", "Unknown"),
            "title": node.get("title", ""),
            "inputs": {},
            "outputs": {},
            "connected_inputs": [],
            "connected_outputs": []
        }
        
        # Collect all inputs
        for idx, input_slot in enumerate(node.get("inputs", [])):
            if input_slot:  # Some inputs might be null
                slot_name = input_slot.get("name", f"input_{idx}")
                slot_type = input_slot.get("type", "Unknown")
                nodes_by_id[node_id]["inputs"][slot_name] = {
                    "type": slot_type,
                    "index": idx,
                    "link": input_slot.get("link")
                }
        
        # Collect all outputs
        for idx, output_slot in enumerate(node.get("outputs", [])):
            if output_slot:  # Some outputs might be null
                slot_name = output_slot.get("name", f"output_{idx}")
                slot_type = output_slot.get("type", "Unknown")
                links = output_slot.get("links", [])
                nodes_by_id[node_id]["outputs"][slot_name] = {
                    "type": slot_type,
                    "index": idx,
                    "links": links if links else []
                }
    
    result["nodes"] = nodes_by_id
    
    # Process links
    for link in workflow_data.get("links", []):
        if len(link) >= 6:
            link_id, source_id, source_slot_idx, target_id, target_slot_idx, link_type = link[:6]
            
            # Mark slots as connected
            if source_id in nodes_by_id:
                if source_slot_idx not in nodes_by_id[source_id]["connected_outputs"]:
                    nodes_by_id[source_id]["connected_outputs"].append(source_slot_idx)
            if target_id in nodes_by_id:
                if target_slot_idx not in nodes_by_id[target_id]["connected_inputs"]:
                    nodes_by_id[target_id]["connected_inputs"].append(target_slot_idx)
            
            # Find slot names
            source_slot_name = None
            target_slot_name = None
            
            if source_id in nodes_by_id:
                for name, slot_info in nodes_by_id[source_id]["outputs"].items():
                    if slot_info["index"] == source_slot_idx:
                        source_slot_name = name
                        break
            
            if target_id in nodes_by_id:
                for name, slot_info in nodes_by_id[target_id]["inputs"].items():
                    if slot_info["index"] == target_slot_idx:
                        target_slot_name = name
                        break
            
            result["links"].append({
                "link_id": link_id,
                "source": {
                    "node_id": source_id,
                    "node_type": nodes_by_id[source_id]["type"] if source_id in nodes_by_id else "Unknown",
                    "slot_index": source_slot_idx,
                    "slot_name": source_slot_name or f"output_{source_slot_idx}",
                    "type": link_type
                },
                "target": {
                    "node_id": target_id,
                    "node_type": nodes_by_id[target_id]["type"] if target_id in nodes_by_id else "Unknown",
                    "slot_index": target_slot_idx,
                    "slot_name": target_slot_name or f"input_{target_slot_idx}",
                    "type": link_type
                }
            })
    
    # Find unconnected slots
    for node_id, node_info in nodes_by_id.items():
        # Unconnected inputs
        for slot_name, slot_info in node_info["inputs"].items():
            if slot_info["index"] not in node_info["connected_inputs"]:
                result["unconnected_slots"]["inputs"].append({
                    "node_id": node_id,
                    "node_type": node_info["type"],
                    "slot_name": slot_name,
                    "slot_type": slot_info["type"]
                })
        
        # Unconnected outputs
        for slot_name, slot_info in node_info["outputs"].items():
            if slot_info["index"] not in node_info["connected_outputs"]:
                result["unconnected_slots"]["outputs"].append({
                    "node_id": node_id,
                    "node_type": node_info["type"],
                    "slot_name": slot_name,
                    "slot_type": slot_info["type"]
                })
    
    return result

def main():
    """Main function to collect all workflow link patterns."""
    workflow_files = get_workflow_files()
    
    all_patterns = {
        "workflows": {},
        "summary": {
            "total_workflows": 0,
            "total_links": 0,
            "total_nodes": 0,
            "total_unconnected_inputs": 0,
            "total_unconnected_outputs": 0
        }
    }
    
    for workflow_file in workflow_files:
        workflow_name = workflow_file.stem
        print(f"\nProcessing: {workflow_name}")
        
        try:
            with open(workflow_file, 'r') as f:
                workflow_data = json.load(f)
            
            patterns = extract_node_info(workflow_data, workflow_name)
            all_patterns["workflows"][workflow_name] = patterns
            
            # Update summary
            all_patterns["summary"]["total_workflows"] += 1
            all_patterns["summary"]["total_links"] += len(patterns["links"])
            all_patterns["summary"]["total_nodes"] += len(patterns["nodes"])
            all_patterns["summary"]["total_unconnected_inputs"] += len(patterns["unconnected_slots"]["inputs"])
            all_patterns["summary"]["total_unconnected_outputs"] += len(patterns["unconnected_slots"]["outputs"])
            
            print(f"  - Nodes: {len(patterns['nodes'])}")
            print(f"  - Links: {len(patterns['links'])}")
            print(f"  - Unconnected inputs: {len(patterns['unconnected_slots']['inputs'])}")
            print(f"  - Unconnected outputs: {len(patterns['unconnected_slots']['outputs'])}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Save results
    output_file = Path(__file__).parent / "link_patterns.json"
    with open(output_file, 'w') as f:
        json.dump(all_patterns, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nSummary:")
    for key, value in all_patterns["summary"].items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()