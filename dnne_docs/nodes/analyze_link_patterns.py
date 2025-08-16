#!/usr/bin/env python3
"""
Analyze collected link patterns to identify unique connection types.
Generates a markdown report with all unique patterns.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

def load_patterns(file_path: str = "link_patterns.json") -> Dict:
    """Load the collected link patterns."""
    patterns_file = Path(__file__).parent / file_path
    if not patterns_file.exists():
        print(f"Patterns file not found: {patterns_file}")
        print("Please run collect_workflow_links.py first")
        return {}
    
    with open(patterns_file, 'r') as f:
        return json.load(f)

def extract_unique_connections(patterns: Dict) -> Dict:
    """Extract unique connection patterns across all workflows."""
    # Widget types that should be excluded from unconnected analysis
    WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
    
    unique_connections = defaultdict(set)  # pattern -> set of workflows
    unique_inputs = defaultdict(set)  # (node_type, slot_name, slot_type) -> workflows
    unique_outputs = defaultdict(set)  # (node_type, slot_name, slot_type) -> workflows
    
    for workflow_name, workflow_data in patterns.get("workflows", {}).items():
        # Process connected links
        for link in workflow_data.get("links", []):
            source = link["source"]
            target = link["target"]
            
            # Create a pattern signature
            pattern = (
                source["node_type"],
                source["slot_name"],
                source["type"],
                target["node_type"],
                target["slot_name"],
                target["type"]
            )
            unique_connections[pattern].add(workflow_name)
        
        # Process unconnected inputs (filter out widgets)
        for slot in workflow_data.get("unconnected_slots", {}).get("inputs", []):
            if slot["slot_type"] not in WIDGET_TYPES:
                pattern = (slot["node_type"], slot["slot_name"], slot["slot_type"])
                unique_inputs[pattern].add(workflow_name)
        
        # Process unconnected outputs (filter out widgets - though outputs shouldn't have widgets)
        for slot in workflow_data.get("unconnected_slots", {}).get("outputs", []):
            if slot["slot_type"] not in WIDGET_TYPES:
                pattern = (slot["node_type"], slot["slot_name"], slot["slot_type"])
                unique_outputs[pattern].add(workflow_name)
    
    return {
        "connections": unique_connections,
        "unconnected_inputs": unique_inputs,
        "unconnected_outputs": unique_outputs
    }

def categorize_connections(unique_patterns: Dict) -> Dict:
    """Categorize connections by their apparent purpose."""
    categories = {
        "layer_connections": [],  # Layer to layer connections
        "data_flow": [],          # Main data tensors
        "training_flow": [],      # Training-related connections
        "control_flow": [],       # Triggers, sync signals
        "config_connections": [], # Configuration connections
        "other": []              # Everything else
    }
    
    for pattern, workflows in unique_patterns["connections"].items():
        src_node, src_slot, src_type, tgt_node, tgt_slot, tgt_type = pattern
        
        # Categorize based on node types and slot names
        if "Layer" in src_node and "Layer" in tgt_node:
            categories["layer_connections"].append((pattern, workflows))
        elif "layers" in src_slot.lower() or "to_output" in tgt_slot.lower():
            categories["layer_connections"].append((pattern, workflows))
        elif any(x in src_type.upper() for x in ["SYNC", "TRIGGER"]):
            categories["control_flow"].append((pattern, workflows))
        elif any(x in src_node for x in ["Dataset", "Sampler", "Batch"]):
            categories["data_flow"].append((pattern, workflows))
        elif any(x in src_node for x in ["Loss", "Optimizer", "Training"]):
            categories["training_flow"].append((pattern, workflows))
        elif "CONFIG" in src_type.upper():
            categories["config_connections"].append((pattern, workflows))
        else:
            categories["other"].append((pattern, workflows))
    
    return categories

def suggest_type_with_format(type_name: str, slot_name: str = "", node_type: str = "", is_output: bool = False) -> str:
    """Suggest a more specific type name with format suffix.
    
    New rules:
    - Outputs are specific (describe what they produce)
    - Inputs are permissive (use wildcards to accept broader types)
    - Exception: passthrough outputs remain generic
    """
    
    # Special handling for Network node
    if node_type == "Network":
        if slot_name == "input":
            return "*TENSOR"  # Input accepts any tensor
        elif slot_name == "output":
            return "NETWORK_OUTPUT_TENSOR" if is_output else "*TENSOR"
        elif slot_name == "to_output":
            return "*LAYER_TENSOR"  # Input that accepts layer tensors
        elif slot_name == "layers":
            return "LAYER_TENSOR" if is_output else "*LAYER_TENSOR"
        elif slot_name == "model":
            return "NETWORK_MODEL" if is_output else "*MODEL"
    
    # Special handling for LinearLayer
    if node_type == "LinearLayer":
        if slot_name == "input":
            return "*LAYER_TENSOR"  # Input accepts layer tensors
        elif slot_name == "output":
            return "LAYER_TENSOR"  # Output produces layer tensor
    
    # GetBatch specific
    if node_type == "GetBatch":
        if slot_name == "labels" and is_output:
            return "BATCH_LABEL_TENSOR"
        elif slot_name == "images" and is_output:
            return "BATCH_IMAGE_TENSOR"
        elif slot_name == "epoch_stats" and is_output:
            return "BATCH_EPOCH_STATS"
        elif slot_name == "trigger" and not is_output:
            return "*TRIGGER"
        elif slot_name == "dataloader" and not is_output:
            return "*BATCH_DATALOADER"
        elif slot_name == "schema" and not is_output:
            return "*BATCH_SCHEMA"
    
    # CrossEntropyLoss
    if node_type == "CrossEntropyLoss":
        if slot_name == "labels" and not is_output:
            return "*LABEL_TENSOR"
        elif slot_name == "predictions" and not is_output:
            return "*TENSOR"
        elif slot_name == "targets" and not is_output:
            return "*TENSOR"  # Alternative input for targets
        elif slot_name == "loss" and is_output:
            return "CROSSENTROPY_LOSS_TENSOR"
        elif slot_name == "accuracy" and is_output:
            return "CROSSENTROPY_ACCURACY_FLOAT"
    
    # TrainingStep
    if node_type == "TrainingStep":
        if slot_name == "ready" and is_output:
            return "TRAIN_STEP_DONE_TRIGGER"
        elif slot_name == "trigger" and is_output:
            return "TRAIN_STEP_TRIGGER"
        elif slot_name == "loss" and not is_output:
            return "*LOSS_TENSOR"
        elif slot_name == "optimizer" and not is_output:
            return "*OPTIMIZER"
    
    # BalancerNode
    if node_type == "BalancerNode":
        if slot_name == "input":
            return "*"  # Accepts anything
        elif slot_name == "output":
            return "*"  # Passthrough - remains wildcard
    
    # IsaacGymSim
    if node_type == "IsaacGymSim":
        if slot_name == "done" and is_output:
            return "SIM_DONE_TRIGGER"
        elif slot_name == "observation" and is_output:
            return "SIM_OBSERVATION_TENSOR"
        elif slot_name == "action" and not is_output:
            return "*TENSOR"
        elif slot_name == "reset" and not is_output:
            return "*TRIGGER"
        elif slot_name == "env_config" and not is_output:
            return "ISAAC_ENV_CONFIG"
    
    # DataStreamer
    if node_type == "DataStreamer":
        if slot_name == "data" and is_output:
            return "DATASTREAMER_DATA_TENSOR"
        elif slot_name == "reset" and not is_output:
            return "*TRIGGER"
        elif slot_name == "sync" and not is_output:
            return "*TRIGGER"
        elif slot_name == "done" and is_output:
            return "DATASTREAMER_DONE_TRIGGER"
        elif slot_name == "metadata" and is_output:
            return "DATASTREAMER_METADATA"
    
    # CustomComputation
    if node_type == "CustomComputation":
        if slot_name == "input" and not is_output:
            return "*TENSOR"
        elif slot_name == "output" and is_output:
            return "CUSTOMCOMP_OUTPUT_TENSOR"
    
    # SGDOptimizer
    if node_type == "SGDOptimizer":
        if slot_name == "model" and not is_output:
            return "NETWORK_MODEL"  # Expects specifically Network's model
        elif slot_name == "optimizer" and is_output:
            return "SGD_OPTIMIZER_OBJ"
    
    # EpochTracker inputs are permissive
    if node_type == "EpochTracker":
        if slot_name == "epoch_stats" and not is_output:
            return "*EPOCH_STATS"
        elif slot_name == "loss" and not is_output:
            return "*LOSS_TENSOR"
        elif slot_name == "accuracy" and not is_output:
            return "*ACCURACY_FLOAT"
        elif slot_name == "training_stats" and is_output:
            return "EPOCH_TRAINING_STATS"
        elif slot_name == "training_summary" and is_output:
            return "EPOCH_TRAINING_SUMMARY"
    
    # Dataset nodes
    if "Dataset" in node_type:
        if slot_name == "dataset" and is_output:
            return f"{node_type.replace('Dataset', '').upper()}_DATASET"
        elif slot_name == "schema" and is_output:
            return f"{node_type.replace('Dataset', '').upper()}_DATASET_SCHEMA"
    
    # BatchSampler
    if node_type == "BatchSampler":
        if slot_name == "dataset" and not is_output:
            return "*DATASET"
        elif slot_name == "schema" and not is_output:
            return "*DATASET_SCHEMA"
        elif slot_name == "dataloader" and is_output:
            return "SAMPLER_BATCH_DATALOADER"
        elif slot_name == "schema" and is_output:
            return "SAMPLER_BATCH_SCHEMA"
    
    # PPO nodes
    if node_type == "PPOAgent":
        if slot_name == "agent" and is_output:
            return "PPO_AGENT_OBJ"
        elif slot_name == "training_stats" and is_output:
            return "PPO_TRAINING_STATS"
        elif slot_name == "eval_stats" and is_output:
            return "PPO_EVAL_STATS"
        elif slot_name == "balancing_config" and not is_output:
            return "BALANCING_CONFIG"
        elif slot_name == "env_config" and not is_output:
            return "ISAAC_ENV_CONFIG"
        elif slot_name == "ppo_config" and not is_output:
            return "PPO_CONFIG"
    
    # Config nodes - outputs are specific
    if "Config" in node_type and slot_name == "config" and is_output:
        return f"{node_type.replace('Config', '').upper()}_CONFIG"
    
    # IsaacGymEnvs
    if node_type == "IsaacGymEnvs":
        if slot_name == "env" and is_output:
            return "ISAAC_ENV_CONFIG"
        elif slot_name == "custom_config" and not is_output:
            return "*CONFIG"
    
    # Default handling for remaining cases
    
    # Schema types
    if type_name == "SCHEMA":
        if is_output:
            return f"{node_type.upper()}_SCHEMA"
        else:
            return "*SCHEMA"
    
    # Dict types
    if type_name == "DICT":
        if is_output:
            # Make output specific based on node and slot
            prefix = node_type.upper() if node_type else "GENERIC"
            suffix = slot_name.upper() if slot_name else "DICT"
            return f"{prefix}_{suffix}"
        else:
            return "*DICT"
    
    # Object types
    if type_name in ["MODEL", "DATASET", "DATALOADER", "OPTIMIZER", "PPO_AGENT"]:
        if is_output:
            return f"{node_type.upper()}_{type_name}"
        else:
            return f"*{type_name}"
    
    # Signal types
    if type_name in ["SYNC", "TRIGGER"]:
        if is_output:
            return f"{node_type.upper()}_{slot_name.upper()}_SIGNAL"
        else:
            return "*SIGNAL"
    
    # Generic tensor handling
    if type_name == "TENSOR":
        if is_output:
            return f"{node_type.upper()}_{slot_name.upper()}_TENSOR"
        else:
            return "*TENSOR"
    
    # Wildcard
    if type_name == "*":
        return "*"
    
    # Default: return as-is
    return type_name

def generate_markdown_report(patterns: Dict, unique_patterns: Dict, categories: Dict) -> str:
    """Generate a markdown report of the analysis."""
    lines = []
    
    # Header
    lines.append("# Link Pattern Analysis Report")
    lines.append("")
    lines.append("## Summary Statistics")
    lines.append("")
    
    if "summary" in patterns:
        for key, value in patterns["summary"].items():
            lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
    
    lines.append("")
    lines.append(f"- **Unique Connection Patterns**: {len(unique_patterns['connections'])}")
    lines.append(f"- **Unique Unconnected Input Types**: {len(unique_patterns['unconnected_inputs'])}")
    lines.append(f"- **Unique Unconnected Output Types**: {len(unique_patterns['unconnected_outputs'])}")
    
    # Connected patterns by category
    lines.append("")
    lines.append("## Connected Link Patterns by Category")
    
    for category_name, category_patterns in categories.items():
        if category_patterns:
            lines.append("")
            lines.append(f"### {category_name.replace('_', ' ').title()}")
            lines.append("")
            
            for pattern, workflows in sorted(category_patterns):
                src_node, src_slot, src_type, tgt_node, tgt_slot, tgt_type = pattern
                workflow_list = ", ".join(sorted(workflows))
                
                # Get suggested types (source is output, target is input)
                src_suggested = suggest_type_with_format(src_type, src_slot, src_node, is_output=True)
                tgt_suggested = suggest_type_with_format(tgt_type, tgt_slot, tgt_node, is_output=False)
                
                # Use suggested types directly for cleaner display
                lines.append(f"- **{src_node}.{src_slot}** ({src_suggested}) → **{tgt_node}.{tgt_slot}** ({tgt_suggested})")
                lines.append(f"  - Used in: {workflow_list}")
    
    # Unconnected inputs
    lines.append("")
    lines.append("## Unconnected Input Slots")
    lines.append("")
    lines.append("These inputs exist but are not connected in any workflow:")
    lines.append("")
    
    for pattern, workflows in sorted(unique_patterns["unconnected_inputs"].items()):
        node_type, slot_name, slot_type = pattern
        workflow_list = ", ".join(sorted(workflows))
        suggested = suggest_type_with_format(slot_type, slot_name, node_type, is_output=False)
        lines.append(f"- **{node_type}.{slot_name}** ({suggested})")
        lines.append(f"  - Found in: {workflow_list}")
    
    # Unconnected outputs
    lines.append("")
    lines.append("## Unconnected Output Slots")
    lines.append("")
    lines.append("These outputs exist but are not connected in any workflow:")
    lines.append("")
    
    for pattern, workflows in sorted(unique_patterns["unconnected_outputs"].items()):
        node_type, slot_name, slot_type = pattern
        workflow_list = ", ".join(sorted(workflows))
        suggested = suggest_type_with_format(slot_type, slot_name, node_type, is_output=True)
        lines.append(f"- **{node_type}.{slot_name}** ({suggested})")
        lines.append(f"  - Found in: {workflow_list}")
    
    # Type usage analysis
    lines.append("")
    lines.append("## Type Usage Analysis")
    lines.append("")
    
    all_types = set()
    suggested_types = set()
    type_mapping = {}
    
    # Collect all types and their suggestions
    for pattern in unique_patterns["connections"].keys():
        src_node, src_slot, src_type, tgt_node, tgt_slot, tgt_type = pattern
        all_types.add(src_type)
        all_types.add(tgt_type)
        
        src_suggested = suggest_type_with_format(src_type, src_slot, src_node, is_output=True)
        tgt_suggested = suggest_type_with_format(tgt_type, tgt_slot, tgt_node, is_output=False)
        
        suggested_types.add(src_suggested)
        suggested_types.add(tgt_suggested)
        
        if src_type != src_suggested:
            if src_type not in type_mapping:
                type_mapping[src_type] = set()
            type_mapping[src_type].add(src_suggested)
        
        if tgt_type != tgt_suggested:
            if tgt_type not in type_mapping:
                type_mapping[tgt_type] = set()
            type_mapping[tgt_type].add(tgt_suggested)
    
    lines.append("### Types Currently in Use")
    lines.append("")
    for type_name in sorted(all_types):
        lines.append(f"- {type_name}")
    
    lines.append("")
    lines.append("### Suggested Type Refinements")
    lines.append("")
    for old_type in sorted(type_mapping.keys()):
        new_types = sorted(type_mapping[old_type])
        lines.append(f"- **{old_type}** → {', '.join(new_types)}")
    
    lines.append("")
    lines.append("### Types in Use after Changes")
    lines.append("")
    
    # Collect all new types after refinement (excluding wildcards)
    new_types_set = set()
    
    # Add all suggested types from connections
    for pattern in unique_patterns["connections"].keys():
        src_node, src_slot, src_type, tgt_node, tgt_slot, tgt_type = pattern
        src_suggested = suggest_type_with_format(src_type, src_slot, src_node, is_output=True)
        tgt_suggested = suggest_type_with_format(tgt_type, tgt_slot, tgt_node, is_output=False)
        # Only add non-wildcard types
        if not src_suggested.startswith('*'):
            new_types_set.add(src_suggested)
        if not tgt_suggested.startswith('*'):
            new_types_set.add(tgt_suggested)
    
    # Add types from unconnected inputs
    for pattern in unique_patterns["unconnected_inputs"].keys():
        node_type, slot_name, slot_type = pattern
        suggested = suggest_type_with_format(slot_type, slot_name, node_type, is_output=False)
        # Only add non-wildcard types
        if not suggested.startswith('*'):
            new_types_set.add(suggested)
    
    # Add types from unconnected outputs
    for pattern in unique_patterns["unconnected_outputs"].keys():
        node_type, slot_name, slot_type = pattern
        suggested = suggest_type_with_format(slot_type, slot_name, node_type, is_output=True)
        # Only add non-wildcard types
        if not suggested.startswith('*'):
            new_types_set.add(suggested)
    
    # Sort and display new types by category
    tensor_types = sorted([t for t in new_types_set if 'TENSOR' in t])
    dict_types = sorted([t for t in new_types_set if 'DICT' in t or 'STATS' in t or 'SUMMARY' in t])
    config_types = sorted([t for t in new_types_set if 'CONFIG' in t])
    obj_types = sorted([t for t in new_types_set if 'OBJ' in t or 'DATASET' in t or 'DATALOADER' in t or 'OPTIMIZER' in t or 'AGENT' in t or 'MODEL' in t])
    signal_types = sorted([t for t in new_types_set if 'TRIGGER' in t or 'SIGNAL' in t])
    schema_types = sorted([t for t in new_types_set if 'SCHEMA' in t])
    other_types = sorted([t for t in new_types_set if t not in tensor_types + dict_types + config_types + obj_types + signal_types + schema_types])
    
    if tensor_types:
        lines.append("#### Tensor Types")
        for t in tensor_types:
            lines.append(f"- {t}")
        lines.append("")
    
    if dict_types:
        lines.append("#### Dictionary/Stats Types")
        for t in dict_types:
            lines.append(f"- {t}")
        lines.append("")
    
    if config_types:
        lines.append("#### Configuration Types")
        for t in config_types:
            lines.append(f"- {t}")
        lines.append("")
    
    if obj_types:
        lines.append("#### Object Types")
        for t in obj_types:
            lines.append(f"- {t}")
        lines.append("")
    
    if signal_types:
        lines.append("#### Signal/Trigger Types")
        for t in signal_types:
            lines.append(f"- {t}")
        lines.append("")
    
    if schema_types:
        lines.append("#### Schema Types")
        for t in schema_types:
            lines.append(f"- {t}")
        lines.append("")
    
    if other_types:
        lines.append("#### Other Types")
        for t in other_types:
            lines.append(f"- {t}")
    
    return "\n".join(lines)

def main():
    """Main analysis function."""
    print("Loading collected patterns...")
    patterns = load_patterns()
    
    if not patterns:
        return
    
    print("Extracting unique patterns...")
    unique_patterns = extract_unique_connections(patterns)
    
    print("Categorizing connections...")
    categories = categorize_connections(unique_patterns)
    
    print("Generating report...")
    report = generate_markdown_report(patterns, unique_patterns, categories)
    
    # Save report
    output_file = Path(__file__).parent / "link_analysis.md"
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {output_file}")
    
    # Print summary
    print("\nQuick Summary:")
    print(f"  Unique connection patterns: {len(unique_patterns['connections'])}")
    print(f"  Unique unconnected inputs: {len(unique_patterns['unconnected_inputs'])}")
    print(f"  Unique unconnected outputs: {len(unique_patterns['unconnected_outputs'])}")
    
    print("\nCategories:")
    for category, items in categories.items():
        if items:
            print(f"  {category}: {len(items)} patterns")

if __name__ == "__main__":
    main()