#!/usr/bin/env python3
"""
Exporter for SplitNode using queue-based template
"""

from ..graph_exporter import ExportableNode
from ..subsystems import SUBSYSTEM_UTIL

class SplitExporter(ExportableNode):
    """Exporter for the Split tensor splitting node"""
    
    @classmethod
    def get_template_name(cls):
        return "nodes/split_node_queue.tpl"
    
    @classmethod
    def parse_split_positions(cls, split_pos, observation_schema, node_id):
        """Parse split positions string and resolve to ranges.
        
        Returns:
            List of tuples: [(start, end), ...] where end is exclusive
        """
        import re
        
        name_list = [x.strip() for x in split_pos.split(',') if x.strip()]
        if not name_list:
            raise ValueError(
                f"SplitNode {node_id}: split_pos '{split_pos}' resulted in empty list"
            )
        
        split_ranges = []
        
        for entry in name_list:
            # Parse pattern: name[slice] or just name
            match = re.match(r'^(\w+)(?:\[([^\]]*)\])?$', entry.strip())
            if not match:
                raise ValueError(
                    f"SplitNode {node_id}: Invalid syntax '{entry}'. "
                    f"Expected format: 'name' or 'name[slice]' (e.g., 'joint_positions[2:5]')"
                )
            
            base_name = match.group(1)
            slice_str = match.group(2)  # May be None if no brackets
            
            # Look up base name in schema
            if base_name not in observation_schema:
                available_names = list(observation_schema.keys())
                raise ValueError(
                    f"SplitNode {node_id}: Unknown semantic name '{base_name}'. "
                    f"Available names: {', '.join(available_names)}"
                )
            
            # Get the base range for this name (INCLUSIVE ranges in schema)
            base_range = observation_schema[base_name]
            
            # Handle both array format [start, end] and single number format
            if isinstance(base_range, (int, float)):
                # Single element: number represents both start and end (inclusive)
                start_idx = int(base_range)
                end_idx_inclusive = int(base_range)
            elif isinstance(base_range, list) and len(base_range) == 2:
                # Array format [start, end] (inclusive)
                start_idx, end_idx_inclusive = base_range
            else:
                raise ValueError(
                    f"SplitNode {node_id}: Invalid range for '{base_name}': {base_range}. "
                    f"Expected [start, end] format or single number."
                )
            
            # Schema uses inclusive ranges, so [0, 6] means indices 0-6 inclusive
            end_idx = end_idx_inclusive + 1  # Convert to exclusive for Python range
            
            # Apply slice if specified
            if slice_str is not None:
                # Parse the slice notation with INCLUSIVE end semantics
                if ':' in slice_str:
                    # It's a slice like [2:5] meaning elements 2,3,4,5 (inclusive)
                    parts = slice_str.split(':')
                    slice_start = int(parts[0]) if parts[0] else None
                    # For inclusive end, add 1 to the stop value
                    slice_stop = (int(parts[1]) + 1) if len(parts) > 1 and parts[1] else None
                    slice_step = int(parts[2]) if len(parts) > 2 and parts[2] else None
                else:
                    # Single index like [3] - extracts just element 3
                    try:
                        single_idx = int(slice_str)
                        slice_start = single_idx
                        slice_stop = single_idx + 1  # Extract single element
                        slice_step = None
                    except ValueError:
                        raise ValueError(
                            f"SplitNode {node_id}: Invalid slice notation '{slice_str}'. "
                            f"Expected integer or slice format (e.g., '3', '2:5', ':3', '::2')"
                        )
                
                # Apply the slice to the base range
                full_range = list(range(start_idx, end_idx))
                if slice_step:
                    sliced_indices = full_range[slice_start:slice_stop:slice_step]
                else:
                    sliced_indices = full_range[slice_start:slice_stop]
                
                if not sliced_indices:
                    raise ValueError(
                        f"SplitNode {node_id}: Slice '{slice_str}' on '{base_name}' "
                        f"resulted in empty selection from range [{start_idx}:{end_idx})"
                    )
                
                # Convert back to range format (start, end exclusive)
                final_start = sliced_indices[0]
                final_end = sliced_indices[-1] + 1
            else:
                # No slice specified - use the entire base range
                final_start = start_idx
                final_end = end_idx
            
            split_ranges.append((final_start, final_end))
        
        return split_ranges
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Extract parameters using helper functions
        param_specs = [
            {'name': 'dimension', 'widget_index': 0},
            {'name': 'split_mode', 'widget_index': 1},
            {'name': 'split_pos', 'widget_index': 2}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Direct access - will fail fast if missing
        dimension = params['dimension']
        split_mode = params['split_mode']
        split_pos = params['split_pos']
        
        # CRITICAL: Per tensor dimension standards in CLAUDE.md:
        # Split MUST operate on dim=1 (features), never dim=0 (batch)
        # This overrides any UI configuration until UI is updated
        dimension = 1  # ALWAYS feature dimension, regardless of split_mode
        
        # Handle different split modes
        if split_mode == "by name":
            # Get input schema to resolve names
            input_schema = cls.get_input_schema(node_data, connections, node_registry, all_nodes, all_links)
            
            # Check if we have schema information
            if not input_schema or 'input' not in input_schema:
                raise ValueError(
                    f"SplitNode {node_id}: No input schema available for 'by name' mode. "
                    f"Make sure node is connected to a source that provides schema (e.g., IsaacGymSim)."
                )
            
            # Get the observation schema if available
            input_info = input_schema['input']
            observation_schema = None
            
            # The schema might be nested in different ways depending on the upstream node
            if isinstance(input_info, dict):
                observation_schema = input_info.get('observation_schema')
            
            if not observation_schema:
                raise ValueError(
                    f"SplitNode {node_id}: No observation_schema found in upstream connection. "
                    f"Available schema keys: {list(input_info.keys()) if isinstance(input_info, dict) else 'none'}"
                )
            
            # Use refactored method to parse split positions
            split_ranges = cls.parse_split_positions(split_pos, observation_schema, node_id)
            
            # Convert tuples to lists for template
            split_values = [[start, end] for start, end in split_ranges]
            # Keep as "by name" but with resolved ranges
            
        else:
            # Original parsing for "by index" and "by size" modes
            try:
                split_values = [int(x.strip()) for x in split_pos.split(',') if x.strip()]
            except ValueError as e:
                raise ValueError(
                    f"SplitNode {node_id}: Failed to parse split_pos '{split_pos}' as comma-separated integers: {e}"
                )
            
            if not split_values:
                raise ValueError(
                    f"SplitNode {node_id}: split_pos '{split_pos}' resulted in empty list"
                )
            
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "SplitNode",
            "DIMENSION": dimension,
            "SPLIT_MODE": split_mode,
            "SPLIT_VALUES": split_values
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import asyncio",
            "from typing import Dict, Any, Optional, List",
        ]
    
    @classmethod
    def get_input_names(cls):
        return ["input"]
    
    @classmethod
    def get_output_names(cls):
        return ["output_a", "output_b", "output_c", "output_d"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        """Return initial schema for split outputs"""
        # Basic schema - output sizes will depend on split configuration
        return {
            "outputs": {
                "output_a": {"type": "tensor", "dtype": "float32", "flattened_size": None},
                "output_b": {"type": "tensor", "dtype": "float32", "flattened_size": None},
                "output_c": {"type": "tensor", "dtype": "float32", "flattened_size": None},
                "output_d": {"type": "tensor", "dtype": "float32", "flattened_size": None}
            }
        }
    
    @classmethod
    def get_output_schema(cls, node_data, connections=None, node_registry=None,
                         all_nodes=None, all_links=None):
        """Get output schema, resolving split output sizes"""
        # Get base schema
        schema = cls.get_initial_output_schema(node_data)
        
        # Extract parameters using helper functions (GOOD - following rules)
        param_specs = [
            {'name': 'split_mode', 'widget_index': 1},
            {'name': 'split_pos', 'widget_index': 2}
        ]
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        split_mode = params['split_mode']
        split_pos = params['split_pos']
        
        if split_mode == "by name" and connections and node_registry and all_nodes and all_links:
            # Get input schema to resolve names
            input_schema = cls.get_input_schema(node_data, connections, node_registry, all_nodes, all_links)
            
            # FAIL-FAST: No fallbacks, require schema
            if not input_schema or 'input' not in input_schema:
                raise ValueError(
                    f"SplitNode: No input schema available for output size calculation"
                )
            
            input_info = input_schema['input']
            observation_schema = input_info.get('observation_schema') if isinstance(input_info, dict) else None
            
            # FAIL-FAST: Require observation schema for "by name" mode
            if not observation_schema:
                raise ValueError(
                    f"SplitNode: No observation_schema found in input connection for 'by name' mode"
                )
            
            # Use refactored method to parse and get ranges
            node_id = str(node_data.get('id', ''))
            split_ranges = cls.parse_split_positions(split_pos, observation_schema, node_id)
            
            # Calculate size for each output
            output_names = ["output_a", "output_b", "output_c", "output_d"]
            for i, (start, end) in enumerate(split_ranges):
                if i < len(output_names):
                    # Size is end - start (since end is exclusive)
                    size = end - start
                    schema["outputs"][output_names[i]]["flattened_size"] = size
        
        return schema

    @classmethod
    def get_subsystem(cls):
        return SUBSYSTEM_UTIL