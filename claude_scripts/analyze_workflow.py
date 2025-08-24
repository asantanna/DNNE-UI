#!/usr/bin/env python
"""
Workflow Analyzer Tool - Optimized for Claude's analysis needs
Provides multiple views and formats of DNNE workflows for easy understanding and debugging
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Any

class WorkflowAnalyzer:
    def __init__(self, workflow_name: str, workflows_dir: str = None):
        # Handle both full path and workflow name
        if "/" in workflow_name:
            # Full path provided
            self.workflow_path = Path(workflow_name)
            self.workflow_name = self.workflow_path.stem
            self.workflows_dir = self.workflow_path.parent
        else:
            # Just workflow name provided
            self.workflow_name = workflow_name.replace(".json", "")
            self.workflows_dir = Path(workflows_dir or "/home/asantanna/DNNE/DNNE-UI/user/default/workflows")
            self.workflow_path = self.workflows_dir / f"{self.workflow_name}.json"
        
        # Create a dedicated directory for this workflow's analysis
        self.output_dir = Path(f"/tmp/{self.workflow_name}_workflow_analysis")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.workflow_data = None
        self.nodes = {}
        self.links = []
        self.error_count = 0
        self.errors = []
        
    def load_workflow(self):
        """Load the workflow JSON file"""
        if not self.workflow_path.exists():
            print(f"❌ Workflow not found: {self.workflow_path}")
            sys.exit(1)
            
        with open(self.workflow_path, 'r') as f:
            self.workflow_data = json.load(f)
            
        # Handle both list and dict formats for nodes
        nodes_data = self.workflow_data.get("nodes", {})
        if isinstance(nodes_data, list):
            # Convert list format to dict format using node's id field
            self.nodes = {str(node.get("id", i)): node for i, node in enumerate(nodes_data)}
        else:
            self.nodes = nodes_data
            
        self.links = self.workflow_data.get("links", [])
        print(f"✓ Loaded {self.workflow_name}: {len(self.nodes)} nodes, {len(self.links)} links")
        
    def create_quick_reference(self) -> str:
        """Create a compact reference format optimized for Claude"""
        lines = []
        lines.append(f"WORKFLOW: {self.workflow_name}")
        lines.append("=" * 60)
        lines.append(f"Total Nodes: {len(self.nodes)} | Total Links: {len(self.links)}\n")
        
        # Build connection map
        connections_out = defaultdict(list)  # node_id -> [(output_slot, target_id, target_slot)]
        connections_in = defaultdict(list)   # node_id -> [(source_id, source_slot, input_slot)]
        
        for link in self.links:
            source_id = str(link[1])
            source_slot = link[2]
            target_id = str(link[3])
            target_slot = link[4]
            connections_out[source_id].append((source_slot, target_id, target_slot))
            connections_in[target_id].append((source_id, source_slot, target_slot))
        
        lines.append("NODE REFERENCE")
        lines.append("-" * 40)
        
        # Sort nodes by ID for consistent ordering
        sorted_nodes = sorted(self.nodes.items(), key=lambda x: int(x[0]))
        
        for node_id, node_data in sorted_nodes:
            # Handle different node formats
            if "class_type" in node_data:
                # ComfyUI format
                class_type = node_data.get("class_type", "Unknown")
                title = node_data.get("_meta", {}).get("title", "")
                inputs = node_data.get("inputs", {})
            else:
                # LiteGraph format
                class_type = node_data.get("type", "Unknown")
                title = node_data.get("title", "")
                inputs = {}
                # Convert inputs list to dict
                for inp in node_data.get("inputs", []):
                    if inp.get("link"):
                        inputs[inp.get("name", "")] = [inp.get("link"), 0]
            
            # Format: Node[ID]: Type (Title) 
            title_str = f" ({title})" if title else ""
            lines.append(f"\nNode[{node_id}]: {class_type}{title_str}")
            
            # Add critical widget values for key node types
            if "widgets_values" in node_data and node_data["widgets_values"]:
                widgets = node_data["widgets_values"]
                if class_type == "Split" and len(widgets) > 2:
                    lines.append(f"  📍 SPLITS: {widgets[2]}")
                elif class_type == "Concat" and len(widgets) > 0:
                    lines.append(f"  📍 MODE: {widgets[0]}")
                elif class_type == "LinearLayer" and len(widgets) > 0:
                    lines.append(f"  📍 SIZE: {widgets[0]} neurons")
                elif class_type == "CustomComputation" and len(widgets) > 0:
                    lines.append(f"  📍 SCRIPT: {widgets[0]}")
                elif class_type in ["IsaacGymEnvs", "IsaacGymSim"] and len(widgets) > 0:
                    task_name = widgets[0] if widgets else 'N/A'
                    lines.append(f"  📍 TASK: {task_name}")
            
            # Show inputs
            if inputs:
                input_list = []
                for inp_name, inp_value in inputs.items():
                    if isinstance(inp_value, list) and len(inp_value) == 2:
                        # This is a connection from another node
                        source_node_id = str(inp_value[0])
                        input_list.append(f"{inp_name}←[{source_node_id}]")
                    else:
                        # This is a direct value
                        input_list.append(f"{inp_name}={repr(inp_value)[:30]}")
                if input_list:
                    lines.append(f"  IN: {', '.join(input_list)}")
            
            # Show outputs (who connects to this node)
            if node_id in connections_out:
                output_list = []
                for slot, target_id, target_slot in connections_out[node_id]:
                    output_list.append(f"→[{target_id}]")
                lines.append(f"  OUT: {', '.join(set(output_list))}")
        
        return "\n".join(lines)
    
    def analyze_data_flow(self) -> str:
        """Trace data flow paths through the workflow"""
        lines = []
        lines.append("DATA FLOW ANALYSIS")
        lines.append("=" * 60)
        
        # Find entry points (nodes with no incoming connections from other nodes)
        nodes_with_inputs = set()
        for link in self.links:
            nodes_with_inputs.add(str(link[3]))
        
        entry_nodes = set(self.nodes.keys()) - nodes_with_inputs
        lines.append(f"\nEntry Points ({len(entry_nodes)} nodes):")
        for node_id in sorted(entry_nodes, key=int):
            node_data = self.nodes[node_id]
            if "class_type" in node_data:
                node_type = node_data.get("class_type", "Unknown")
                title = node_data.get("_meta", {}).get("title", "")
            else:
                node_type = node_data.get("type", "Unknown")
                title = node_data.get("title", "")
            title_str = f" - {title}" if title else ""
            lines.append(f"  [{node_id}] {node_type}{title_str}")
        
        # Find terminal nodes (nodes with no outgoing connections)
        nodes_with_outputs = set()
        for link in self.links:
            nodes_with_outputs.add(str(link[1]))
        
        terminal_nodes = set(self.nodes.keys()) - nodes_with_outputs
        lines.append(f"\nTerminal Nodes ({len(terminal_nodes)} nodes):")
        for node_id in sorted(terminal_nodes, key=int):
            node_data = self.nodes[node_id]
            if "class_type" in node_data:
                node_type = node_data.get("class_type", "Unknown")
                title = node_data.get("_meta", {}).get("title", "")
            else:
                node_type = node_data.get("type", "Unknown")
                title = node_data.get("title", "")
            title_str = f" - {title}" if title else ""
            lines.append(f"  [{node_id}] {node_type}{title_str}")
        
        # Trace paths from entry to terminal nodes
        lines.append("\nMain Data Paths:")
        paths = self._find_paths(entry_nodes, terminal_nodes)
        for i, path in enumerate(paths[:10], 1):  # Show first 10 paths
            path_str = " → ".join([f"[{nid}]{self.nodes[nid].get('class_type', 'Unknown')}" for nid in path])
            lines.append(f"  Path {i}: {path_str}")
        
        if len(paths) > 10:
            lines.append(f"  ... and {len(paths)-10} more paths")
        
        return "\n".join(lines)
    
    def _find_paths(self, start_nodes: Set[str], end_nodes: Set[str]) -> List[List[str]]:
        """Find paths from start nodes to end nodes"""
        # Build adjacency list
        graph = defaultdict(list)
        for link in self.links:
            source = str(link[1])
            target = str(link[3])
            graph[source].append(target)
        
        paths = []
        for start in start_nodes:
            for end in end_nodes:
                # BFS to find path
                queue = deque([(start, [start])])
                visited = set()
                
                while queue and len(paths) < 20:  # Limit total paths
                    node, path = queue.popleft()
                    
                    if node == end:
                        paths.append(path)
                        continue
                    
                    if node in visited:
                        continue
                    visited.add(node)
                    
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            queue.append((neighbor, path + [neighbor]))
        
        return paths
    
    def create_dependency_graph(self) -> Dict:
        """Create a dependency graph showing execution order"""
        deps = {}
        
        # Build dependency information
        for node_id in self.nodes:
            node_data = self.nodes[node_id]
            if "class_type" in node_data:
                node_type = node_data.get("class_type", "Unknown")
                title = node_data.get("_meta", {}).get("title", "")
            else:
                node_type = node_data.get("type", "Unknown")
                title = node_data.get("title", "")
            
            deps[node_id] = {
                "type": node_type,
                "title": title,
                "depends_on": [],
                "required_by": [],
                "level": -1  # Execution level (0 = entry point)
            }
        
        # Fill in dependencies from links
        for link in self.links:
            source = str(link[1])
            target = str(link[3])
            deps[target]["depends_on"].append(source)
            deps[source]["required_by"].append(target)
        
        # Calculate execution levels
        # Level 0 = nodes with no dependencies
        queue = deque()
        for node_id, info in deps.items():
            if not info["depends_on"]:
                info["level"] = 0
                queue.append(node_id)
        
        # BFS to assign levels
        while queue:
            node_id = queue.popleft()
            current_level = deps[node_id]["level"]
            
            for required_node in deps[node_id]["required_by"]:
                if deps[required_node]["level"] == -1:
                    # Check if all dependencies have been assigned levels
                    all_deps_ready = all(
                        deps[dep]["level"] >= 0 
                        for dep in deps[required_node]["depends_on"]
                    )
                    if all_deps_ready:
                        max_dep_level = max(
                            deps[dep]["level"] 
                            for dep in deps[required_node]["depends_on"]
                        )
                        deps[required_node]["level"] = max_dep_level + 1
                        queue.append(required_node)
        
        return deps
    
    def analyze_split_concat_patterns(self) -> str:
        """Analyze Split and Concat patterns - crucial for understanding data flow."""
        lines = []
        lines.append("SPLIT/CONCAT PATTERN ANALYSIS")
        lines.append("=" * 60)
        
        splits = []
        concats = []
        
        for node_id, node_data in self.nodes.items():
            if "class_type" in node_data:
                node_type = node_data.get("class_type", "")
            else:
                node_type = node_data.get("type", "")
                
            widgets = node_data.get("widgets_values", [])
            
            if node_type == "Split":
                split_mode = widgets[1] if len(widgets) > 1 else "unknown"
                split_fields = widgets[2] if len(widgets) > 2 else "unknown"
                splits.append({
                    "id": node_id,
                    "mode": split_mode,
                    "fields": split_fields
                })
            elif node_type == "Concat":
                concat_mode = widgets[0] if len(widgets) > 0 else "unknown"
                pad_mode = widgets[1] if len(widgets) > 1 else "unknown"
                concats.append({
                    "id": node_id,
                    "mode": concat_mode,
                    "pad": pad_mode
                })
        
        if splits:
            lines.append("\nSplit Nodes (Data Distribution):")
            for s in sorted(splits, key=lambda x: int(x["id"])):
                lines.append(f"  Node[{s['id']}]:")
                lines.append(f"    Mode: {s['mode']}")
                lines.append(f"    Fields: {s['fields']}")
                # Show what connects to this split
                for link in self.links:
                    if str(link[3]) == s['id']:
                        source_id = str(link[1])
                        source_node = self.nodes.get(source_id, {})
                        source_type = source_node.get("class_type", source_node.get("type", "Unknown"))
                        lines.append(f"    ← From: [{source_id}] {source_type}")
        
        if concats:
            lines.append("\nConcat Nodes (Data Aggregation):")
            for c in sorted(concats, key=lambda x: int(x["id"])):
                lines.append(f"  Node[{c['id']}]:")
                lines.append(f"    Mode: {c['mode']}")
                lines.append(f"    Padding: {c['pad']}")
                # Show what this concat feeds into
                for link in self.links:
                    if str(link[1]) == c['id']:
                        target_id = str(link[3])
                        target_node = self.nodes.get(target_id, {})
                        target_type = target_node.get("class_type", target_node.get("type", "Unknown"))
                        lines.append(f"    → To: [{target_id}] {target_type}")
        
        return "\n".join(lines)
    
    def identify_node_clusters(self) -> Dict[str, List[str]]:
        """Group nodes by functionality based on type patterns"""
        clusters = defaultdict(list)
        
        for node_id, node_data in self.nodes.items():
            class_type = node_data.get("class_type", node_data.get("type", "Unknown"))
            
            # Categorize based on node type
            if "Isaac" in class_type or "Gym" in class_type:
                clusters["Environment"].append(node_id)
            elif "PPO" in class_type or "Agent" in class_type:
                clusters["Agent"].append(node_id)
            elif "Training" in class_type or "Optimizer" in class_type or "Loss" in class_type:
                clusters["Training"].append(node_id)
            elif "Network" in class_type or "Linear" in class_type or "Conv" in class_type:
                clusters["Network"].append(node_id)
            elif "Tensor" in class_type or "Concat" in class_type or "Split" in class_type:
                clusters["Tensor Operations"].append(node_id)
            elif "Dataset" in class_type or "Batch" in class_type:
                clusters["Data"].append(node_id)
            elif "Config" in class_type or "Setting" in class_type:
                clusters["Configuration"].append(node_id)
            elif "Yield" in class_type or "Queue" in class_type:
                clusters["Control Flow"].append(node_id)
            else:
                clusters["Other"].append(node_id)
        
        # Sort node IDs in each cluster
        for cluster_name in clusters:
            clusters[cluster_name].sort(key=int)
        
        return dict(clusters)
    
    def add_error(self, error_msg: str):
        """Add an error to the error list and increment count"""
        self.error_count += 1
        self.errors.append(error_msg)
        
    def analyze_label_connections(self) -> str:
        """Analyze label connections in the workflow"""
        lines = []
        lines.append("LABEL CONNECTION ANALYSIS")
        lines.append("=" * 60)
        
        # Get label dictionary if it exists
        label_dict = self.workflow_data.get("extra", {}).get("labelDictionary", {})
        
        if not label_dict:
            lines.append("\nNo label connections found in this workflow.")
            return "\n".join(lines)
        
        # Count different types of labels
        output_labels = []
        input_labels = []
        dynamic_input_labels = []
        
        for label_name, label_info in label_dict.items():
            if "_input_" in label_name:
                dynamic_input_labels.append(label_name)
            elif label_info.get("direction") == "output":
                output_labels.append(label_name)
            elif label_info.get("direction") == "input":
                input_labels.append(label_name)
        
        lines.append(f"\nLabel Statistics:")
        lines.append(f"  Output labels: {len(output_labels)}")
        lines.append(f"  Input labels: {len(input_labels)}")
        lines.append(f"  Dynamic input labels: {len(dynamic_input_labels)}")
        lines.append("")
        
        # Track which labels are actually used
        used_output_labels = set()
        used_input_labels = set()
        orphaned_labels = []
        mismatched_labels = []
        
        # Check for orphaned Label nodes (nodes with no actual connections)
        nodes = self.workflow_data.get("nodes", [])
        links = self.workflow_data.get("links", [])
        
        # Build connection map
        nodes_with_incoming = set()
        nodes_with_outgoing = set()
        for link in links:
            if len(link) >= 6:
                source_node = str(link[1])
                target_node = str(link[3])
                nodes_with_outgoing.add(source_node)
                nodes_with_incoming.add(target_node)
        
        # Check Label nodes for orphans
        for node in nodes:
            node_id = str(node.get("id"))
            node_type = node.get("type", "")
            
            if node_type == "Label":
                # Get label properties from the properties field (FAIL-FAST)
                properties = node.get("properties")
                if properties is None:
                    error_msg = f"Label node {node_id} missing 'properties' field"
                    self.add_error(error_msg)
                    orphaned_labels.append(error_msg)
                    continue
                    
                label_name = properties.get("labelName")
                if label_name is None:
                    error_msg = f"Label node {node_id} missing 'labelName' in properties"
                    self.add_error(error_msg)
                    orphaned_labels.append(error_msg)
                    continue
                    
                label_direction = properties.get("labelDirection")
                if label_direction is None:
                    error_msg = f"Label node {node_id} missing 'labelDirection' in properties"
                    self.add_error(error_msg)
                    orphaned_labels.append(error_msg)
                    continue
                
                # Check if this Label node has appropriate connections
                # INPUT-direction labels receive from the label system and OUTPUT to other nodes
                # OUTPUT-direction labels receive from other nodes and provide to the label system
                if label_direction == "input" and node_id not in nodes_with_outgoing:
                    # Input direction labels without OUTGOING connections are errors
                    # They receive data from label but don't pass it anywhere
                    error_msg = f"Label node {node_id} (receives '{label_name}') has no outgoing connection"
                    self.add_error(error_msg)
                    orphaned_labels.append(error_msg)
                elif label_direction == "output" and node_id not in nodes_with_incoming:
                    # Output direction labels without INCOMING connections are just unused
                    # They would provide data to label system but have no source
                    orphaned_labels.append(f"Label node {node_id} (provides '{label_name}') has no incoming connection [unused]")
        
        # Check all dynamic input labels to see what they connect to
        for dyn_label in dynamic_input_labels:
            dyn_info = label_dict[dyn_label]
            connected_to = dyn_info.get("connectedToLabel")
            if connected_to:
                if connected_to in label_dict:
                    used_output_labels.add(connected_to)
                else:
                    mismatched_labels.append(f"Dynamic label '{dyn_label}' references non-existent label '{connected_to}'")
        
        # Check which input labels have matching output labels
        for input_label in input_labels:
            # Find matching output label by name
            matching_output = None
            for output_label in output_labels:
                if output_label == input_label or output_label.replace("_output", "") == input_label.replace("_input", ""):
                    matching_output = output_label
                    break
            
            if matching_output:
                used_input_labels.add(input_label)
                used_output_labels.add(matching_output)
            else:
                # Check if any dynamic label connects to this input label
                found_connection = False
                for dyn_label in dynamic_input_labels:
                    if label_dict[dyn_label].get("connectedToLabel") == input_label:
                        found_connection = True
                        used_input_labels.add(input_label)
                        break
                
                if not found_connection:
                    orphaned_labels.append(f"Input label '{input_label}' (Node {label_dict[input_label].get('anchorNodeId')}) has no matching output")
        
        # Find unused output labels
        for output_label in output_labels:
            if output_label not in used_output_labels:
                node_id = label_dict[output_label].get("nodeId")
                anchor_id = label_dict[output_label].get("anchorNodeId")
                orphaned_labels.append(f"Output label '{output_label}' (Node {node_id}, Anchor {anchor_id}) is not connected to any input")
        
        # Report issues - separate errors from warnings
        error_orphans = [o for o in orphaned_labels if "(input)" in o and "[unused]" not in o]
        warning_orphans = [o for o in orphaned_labels if "[unused]" in o]
        
        if error_orphans:
            lines.append("")
            lines.append("❌ ORPHANED INPUT LABELS (ERRORS):")
            lines.append("-" * 40)
            for orphan in error_orphans:
                lines.append(f"  • {orphan}")
            lines.append("")
            lines.append("💡 To repair: Run with --repair-labels to remove orphaned labels")
            lines.append("   (Original workflow will be backed up)")
            lines.append("")
        
        if warning_orphans:
            lines.append("")
            lines.append("⚠️ UNUSED OUTPUT LABELS (not errors):")
            lines.append("-" * 40)
            for orphan in warning_orphans:
                lines.append(f"  • {orphan}")
            lines.append("")
        
        if mismatched_labels:
            lines.append("❌ MISMATCHED LABELS (reference non-existent labels):")
            for mismatch in mismatched_labels:
                lines.append(f"  - {mismatch}")
            lines.append("")
        
        # Detailed analysis of each label pair
        lines.append("DETAILED LABEL CONNECTIONS:")
        lines.append("-" * 40)
        
        # Organize labels by base name
        label_pairs = defaultdict(dict)
        for label_name, label_info in label_dict.items():
            if "_input_" in label_name:
                # This is a dynamic input label
                base_name = label_name.split("_input_")[0]
                label_pairs[base_name]["inputs"] = label_pairs[base_name].get("inputs", [])
                label_pairs[base_name]["inputs"].append({
                    "label": label_name,
                    "node": label_info.get("nodeId"),
                    "slot": label_info.get("slotName"),
                    "connected_to": label_info.get("connectedToLabel")
                })
            else:
                # This is a regular label
                label_pairs[label_name]["info"] = label_info
        
        # Analyze each label pair
        for label_name, label_data in sorted(label_pairs.items()):
            if "info" not in label_data:
                # This is an input-only entry, check if it's orphaned
                if label_data.get("inputs"):
                    lines.append(f"Label Group: {label_name}")
                    lines.append(f"  ⚠️ Has dynamic inputs but no base label definition")
                    for inp in label_data["inputs"]:
                        lines.append(f"    Input: Node[{inp['node']}].{inp['slot']} → {inp.get('connected_to', 'NONE')}")
                    lines.append("")
                continue
                
            info = label_data["info"]
            node_id = info.get("nodeId")
            slot_name = info.get("slotName")
            slot_type = info.get("slotType")
            direction = info.get("direction")
            anchor_id = info.get("anchorNodeId")
            
            # Check if this node actually exists
            node_exists = str(node_id) in self.nodes or str(anchor_id) in self.nodes
            
            lines.append(f"Label: {label_name}")
            if not node_exists:
                lines.append(f"  ❌ ERROR: References non-existent node {node_id} (anchor {anchor_id})")
            lines.append(f"  Source: Node[{node_id}].{slot_name} ({direction})")
            lines.append(f"  Type: {slot_type}")
            lines.append(f"  Anchor: Node[{anchor_id}]")
            
            # Find matching input labels
            inputs = label_data.get("inputs", [])
            if inputs:
                lines.append(f"  Connected to {len(inputs)} input(s):")
                for inp in inputs:
                    lines.append(f"    → Node[{inp['node']}].{inp['slot']}")
            else:
                # Look for direct connections
                connected_count = 0
                for other_name, other_data in label_pairs.items():
                    if "inputs" in other_data:
                        for inp in other_data["inputs"]:
                            if inp.get("connected_to") == label_name:
                                lines.append(f"  Connected to:")
                                lines.append(f"    → Node[{inp['node']}].{inp['slot']}")
                                connected_count += 1
                
                if connected_count == 0 and direction == "output":
                    lines.append(f"  ⚠️ WARNING: Output label not connected to any inputs!")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def check_export_readiness(self) -> Dict:
        """Check if workflow is ready for export"""
        issues = []
        warnings = []
        info = []
        
        # Check for disconnected nodes (excluding Label nodes)
        connected_nodes = set()
        for link in self.links:
            connected_nodes.add(str(link[1]))
            connected_nodes.add(str(link[3]))
        
        # Don't count Label nodes as disconnected
        disconnected = set()
        for node_id in self.nodes.keys():
            if node_id not in connected_nodes:
                node_type = self.nodes[node_id].get("type", self.nodes[node_id].get("class_type", ""))
                if node_type != "Label":
                    disconnected.add(node_id)
        
        if disconnected:
            issues.append(f"Disconnected nodes found: {sorted(disconnected, key=int)}")
        
        # Check for required node types
        node_types = set()
        for node in self.nodes.values():
            node_type = node.get("class_type", node.get("type", ""))
            if node_type:
                node_types.add(node_type)
        
        # Check for cycles
        if self._has_cycle():
            warnings.append("Workflow contains cycles - ensure they're intentional (e.g., training loops)")
        
        # Check for missing connections in node inputs
        for node_id, node_data in self.nodes.items():
            if "class_type" in node_data:
                # ComfyUI format
                inputs = node_data.get("inputs", {})
                for input_name, input_value in inputs.items():
                    if isinstance(input_value, list) and len(input_value) == 2:
                        source_node_id = str(input_value[0])
                        if source_node_id not in self.nodes:
                            issues.append(f"Node[{node_id}] references non-existent node[{source_node_id}]")
            # LiteGraph format validation is handled by link validation above
        
        # Check label connections
        label_dict = self.workflow_data.get("extra", {}).get("labelDictionary", {})
        if label_dict:
            info.append(f"Label connections: {len([k for k in label_dict.keys() if not '_input_' in k])} labels")
            
            # Check for orphaned labels
            for label_name, label_info in label_dict.items():
                if "_input_" in label_name:
                    connected_to = label_info.get("connectedToLabel")
                    if connected_to and connected_to not in label_dict:
                        warnings.append(f"Label input '{label_name}' references non-existent label '{connected_to}'")
        
        # Check for nodes with unconnected inputs (considering both direct and label connections)
        # Define which node types have strictly required inputs (no defaults)
        required_inputs_by_type = {
            "SGDOptimizer": ["model", "loss"],
            "Network": ["input", "to_output"],
            "LinearLayer": ["input"],
            "Concat": [], # Can work with partial inputs
            "Split": ["input"],
            "Barrier": ["input", "release"],
            "CustomComputation": ["input"],
            "SimulationTracker": ["observation", "done"],
            "IsaacGymSim": ["env_config", "action"],
        }
        
        for node_id, node_data in self.nodes.items():
            node_type = node_data.get("type", node_data.get("class_type", ""))
            
            # Skip Label nodes
            if node_type == "Label":
                continue
                
            # Get required inputs for this node type
            required_inputs = []
            for type_pattern, inputs in required_inputs_by_type.items():
                if type_pattern in node_type:
                    required_inputs = inputs
                    break
            
            # Get node inputs
            node_inputs = node_data.get("inputs", [])
            if isinstance(node_inputs, list):
                # LiteGraph format - inputs are a list
                for input_idx, input_info in enumerate(node_inputs):
                    input_name = input_info.get("name", f"input_{input_idx}")
                    input_link = input_info.get("link")
                    input_links = input_info.get("links", [])
                    
                    # Check if this input has no connections (neither single nor multiple)
                    if not input_link and not input_links:
                        # Check if it's connected via a label
                        has_label_connection = False
                        
                        if label_dict:
                            # Look for label connections to this input
                            # Check both direct label inputs and dynamic label inputs
                            for label_name, label_info in label_dict.items():
                                if "_input_" in label_name:
                                    # Dynamic input label (e.g., "Barrier(74).release_input_74_1")
                                    if (str(label_info.get("nodeId")) == node_id and 
                                        label_info.get("slotName") == input_name):
                                        connected_label = label_info.get("connectedToLabel")
                                        if connected_label and connected_label in label_dict:
                                            has_label_connection = True
                                            info.append(f"Node[{node_id}].{input_name} connected via label '{connected_label}'")
                                            break
                        
                        # Check if this is a required input
                        is_required = input_name in required_inputs
                        
                        # Only report if it's not a widget input and has no connection
                        if not input_info.get("widget") and not has_label_connection:
                            if is_required:
                                error_msg = f"Node[{node_id}] {node_type} missing REQUIRED input '{input_name}'"
                                issues.append(error_msg)
                                self.add_error(error_msg)
                            else:
                                # Only warn about non-required inputs if they're not optional
                                # Some inputs like reset, reward, etc. are truly optional
                                optional_inputs = ["reset", "reward", "custom_metrics", "input_d", "input_c"]
                                if input_name not in optional_inputs:
                                    warnings.append(f"Node[{node_id}] {node_type} has unconnected input '{input_name}'")
        
        # Provide statistics
        info.append(f"Total nodes: {len(self.nodes)}")
        info.append(f"Total connections: {len(self.links)}")
        info.append(f"Unique node types: {len(node_types)}")
        
        deps = self.create_dependency_graph()
        max_level = max((d["level"] for d in deps.values()), default=0)
        info.append(f"Maximum execution depth: {max_level}")
        
        return {
            "ready": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "info": info,
            "node_types": sorted(node_types)
        }
    
    def _has_cycle(self) -> bool:
        """Check if the workflow graph has cycles using DFS"""
        graph = defaultdict(list)
        for link in self.links:
            graph[str(link[1])].append(str(link[3]))
        
        visited = set()
        rec_stack = set()
        
        def has_cycle_util(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if has_cycle_util(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.nodes:
            if node not in visited:
                if has_cycle_util(node):
                    return True
        
        return False
    
    def repair_orphaned_labels(self):
        """Remove orphaned label nodes from the workflow"""
        import shutil
        from datetime import datetime
        
        # Create backup
        backup_path = self.workflow_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        shutil.copy2(self.workflow_path, backup_path)
        print(f"✓ Created backup: {backup_path.name}")
        
        # Find orphaned label nodes
        orphaned_nodes = []
        removed_count = 0
        
        # Get label dictionary
        label_dict = self.workflow_data.get("extra", {}).get("labelDictionary", {})
        
        # Find all Label nodes
        for node_id, node_data in self.nodes.items():
            node_type = node_data.get("type", node_data.get("class_type", ""))
            if node_type == "Label":
                # Check if this label node is properly connected
                label_name = node_data.get("properties", {}).get("labelName", "")
                label_direction = node_data.get("properties", {}).get("labelDirection", "")
                
                # Check if it's an anchor node for a label in the dictionary
                is_anchor = False
                for dict_label, dict_info in label_dict.items():
                    if str(dict_info.get("anchorNodeId")) == str(node_id):
                        is_anchor = True
                        break
                
                if is_anchor:
                    continue  # This is a valid anchor node
                
                # Only repair orphaned INPUT-direction labels without outgoing connections
                # These are labels that receive from the label system but don't pass data anywhere
                if label_direction == "input":
                    has_output = False
                    # Check outputs array for outgoing connections
                    outputs = node_data.get("outputs", [])
                    for out in outputs:
                        if out.get("links") and len(out.get("links", [])) > 0:
                            has_output = True
                            break
                    
                    if not has_output:
                        orphaned_nodes.append((node_id, f"Label receiving '{label_name}' has no outgoing connection"))
        
        if not orphaned_nodes:
            print("✓ No orphaned label nodes found")
            return False
        
        print(f"\nFound {len(orphaned_nodes)} orphaned label node(s):")
        for node_id, reason in orphaned_nodes:
            print(f"  - Node[{node_id}]: {reason}")
        
        # Remove orphaned nodes
        print("\nRemoving orphaned nodes...")
        
        # Remove from nodes array
        new_nodes = []
        for node in self.workflow_data.get("nodes", []):
            if str(node.get("id")) not in [str(nid) for nid, _ in orphaned_nodes]:
                new_nodes.append(node)
            else:
                removed_count += 1
                print(f"  ✗ Removed node {node.get('id')}")
        
        self.workflow_data["nodes"] = new_nodes
        
        # Also remove any links that reference these nodes
        removed_links = 0
        if "links" in self.workflow_data:
            new_links = []
            for link in self.workflow_data["links"]:
                # link format: [link_id, source_node, source_slot, target_node, target_slot, type]
                source_node = str(link[1]) if len(link) > 1 else None
                target_node = str(link[3]) if len(link) > 3 else None
                
                if source_node not in [str(nid) for nid, _ in orphaned_nodes] and \
                   target_node not in [str(nid) for nid, _ in orphaned_nodes]:
                    new_links.append(link)
                else:
                    removed_links += 1
            
            self.workflow_data["links"] = new_links
            
            if removed_links > 0:
                print(f"  ✗ Removed {removed_links} link(s) to/from orphaned nodes")
        
        # Save repaired workflow
        with open(self.workflow_path, 'w') as f:
            json.dump(self.workflow_data, f, indent=2)
        
        print(f"\n✓ Repaired workflow saved to: {self.workflow_path}")
        print(f"  Removed {removed_count} orphaned node(s)")
        print(f"  Backup saved as: {backup_path.name}")
        
        return True
    
    def save_all_outputs(self):
        """Save all analysis outputs to files"""
        # All files go in the dedicated directory
        
        # Save full workflow
        with open(self.output_dir / "full.json", 'w') as f:
            json.dump(self.workflow_data, f, indent=2)
        print(f"✓ Full workflow: full.json")
        
        # Save nodes only
        with open(self.output_dir / "nodes.json", 'w') as f:
            json.dump(self.nodes, f, indent=2)
        print(f"✓ Nodes only: nodes.json")
        
        # Save links only
        with open(self.output_dir / "links.json", 'w') as f:
            json.dump(self.links, f, indent=2)
        print(f"✓ Links only: links.json")
        
        # Save quick reference
        quickref = self.create_quick_reference()
        with open(self.output_dir / "quickref.txt", 'w') as f:
            f.write(quickref)
        print(f"✓ Quick reference: quickref.txt")
        
        # Save data flow analysis
        dataflow = self.analyze_data_flow()
        with open(self.output_dir / "dataflow.txt", 'w') as f:
            f.write(dataflow)
        print(f"✓ Data flow: dataflow.txt")
        
        # Save dependency graph
        deps = self.create_dependency_graph()
        with open(self.output_dir / "dependencies.json", 'w') as f:
            json.dump(deps, f, indent=2)
        print(f"✓ Dependencies: dependencies.json")
        
        # Save split/concat analysis (NEW - most useful for complex workflows!)
        split_concat = self.analyze_split_concat_patterns()
        with open(self.output_dir / "split_concat.txt", 'w') as f:
            f.write(split_concat)
        print(f"✓ Split/Concat patterns: split_concat.txt")
        
        # Save label connection analysis
        label_analysis = self.analyze_label_connections()
        with open(self.output_dir / "label_connections.txt", 'w') as f:
            f.write(label_analysis)
        print(f"✓ Label connections: label_connections.txt")
        
        # Save node clusters
        clusters = self.identify_node_clusters()
        with open(self.output_dir / "clusters.json", 'w') as f:
            json.dump(clusters, f, indent=2)
        print(f"✓ Node clusters: clusters.json")
        
        # Save export readiness check
        export_check = self.check_export_readiness()
        with open(self.output_dir / "export_ready.json", 'w') as f:
            json.dump(export_check, f, indent=2)
        print(f"✓ Export check: export_ready.json")
        
        # Create summary report
        summary = []
        summary.append(f"WORKFLOW ANALYSIS SUMMARY: {self.workflow_name}")
        summary.append("=" * 60)
        summary.append(f"\nStatistics:")
        summary.append(f"  Nodes: {len(self.nodes)}")
        summary.append(f"  Links: {len(self.links)}")
        summary.append(f"  Node Types: {len(set(n.get('class_type', '') for n in self.nodes.values()))}")
        
        summary.append(f"\nNode Clusters:")
        for cluster_name, node_ids in clusters.items():
            if node_ids:
                summary.append(f"  {cluster_name}: {len(node_ids)} nodes")
        
        summary.append(f"\nExport Readiness: {'✓ READY' if export_check['ready'] else '✗ NOT READY'}")
        if export_check['issues']:
            summary.append("  Issues:")
            for issue in export_check['issues']:
                summary.append(f"    - {issue}")
        if export_check['warnings']:
            summary.append("  Warnings:")
            for warning in export_check['warnings']:
                summary.append(f"    - {warning}")
        
        with open(self.output_dir / "summary.txt", 'w') as f:
            f.write("\n".join(summary))
        print(f"✓ Summary: summary.txt")
        
        return str(self.output_dir)

def main():
    parser = argparse.ArgumentParser(description="Analyze DNNE workflow structure")
    parser.add_argument("workflow", help="Name of the workflow to analyze (without .json)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--compare", help="Compare with another workflow")
    parser.add_argument("--dir", help="Custom workflows directory")
    parser.add_argument("--repair-labels", action="store_true", 
                       help="Repair workflow by removing orphaned label nodes. Creates a backup before modification. WARNING: NOT TESTED YET!")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"DNNE WORKFLOW ANALYZER")
    print(f"{'='*60}\n")
    
    # Analyze primary workflow
    analyzer = WorkflowAnalyzer(args.workflow, args.dir)
    analyzer.load_workflow()
    prefix = analyzer.save_all_outputs()
    
    if args.verbose:
        print("\n" + analyzer.create_quick_reference())
        print("\n" + analyzer.analyze_data_flow())
    
    # Compare with another workflow if requested
    if args.compare:
        print(f"\n{'='*60}")
        print(f"COMPARING WITH: {args.compare}")
        print(f"{'='*60}\n")
        
        analyzer2 = WorkflowAnalyzer(args.compare, args.dir)
        analyzer2.load_workflow()
        
        # Simple comparison
        print(f"Nodes: {args.workflow}={len(analyzer.nodes)} vs {args.compare}={len(analyzer2.nodes)}")
        print(f"Links: {args.workflow}={len(analyzer.links)} vs {args.compare}={len(analyzer2.links)}")
        
        types1 = set(n.get("class_type", "") for n in analyzer.nodes.values())
        types2 = set(n.get("class_type", "") for n in analyzer2.nodes.values())
        
        only_in_1 = types1 - types2
        only_in_2 = types2 - types1
        
        if only_in_1:
            print(f"\nNode types only in {args.workflow}:")
            for t in sorted(only_in_1):
                print(f"  - {t}")
        
        if only_in_2:
            print(f"\nNode types only in {args.compare}:")
            for t in sorted(only_in_2):
                print(f"  - {t}")
    
    # Handle repair if requested
    if args.repair_labels:
        if analyzer.repair_orphaned_labels():
            print("\n✓ Workflow repaired successfully")
            # Reset error count after successful repair
            analyzer.error_count = 0
            analyzer.errors = []
        else:
            print("\n✓ No orphaned labels found - no repairs needed")
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Files saved to: {prefix}/")
    print(f"{'='*60}")
    
    # Report error summary
    if analyzer.error_count == 0:
        print("\n✅ No errors detected.")
    else:
        print(f"\n❌ {analyzer.error_count} error(s) detected.")
        print(f"\nErrors found:")
        for error in analyzer.errors:
            print(f"  • {error}")
        print(f"\nFor more information see:")
        print(f"  • {prefix}/label_connections.txt - Label connection analysis")
        print(f"  • {prefix}/export_ready.json - Export readiness check")
        if not args.repair_labels:
            print(f"\n💡 To repair label errors: Run with --repair-labels")
    
    print(f"\nKey files to check:")
    print(f"  • {prefix}/quickref.txt - Node reference with key parameters")
    print(f"  • {prefix}/split_concat.txt - Split/Concat patterns")
    print(f"  • {prefix}/summary.txt - Quick overview")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()