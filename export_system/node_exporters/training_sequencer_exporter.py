#!/usr/bin/env python3
"""
Exporter for TrainingSequencer node
"""

from ..graph_exporter import ExportableNode
from ..utils import export_utils

class TrainingSequencerExporter(ExportableNode):
    """Exporter for the Training Sequencer node"""
    
    @classmethod
    def get_template_name(cls):
        return "nodes/training_sequencer_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Extract widget parameters
        param_specs = [
            {'name': 'order', 'widget_index': 0},
            {'name': 'retain_graph', 'widget_index': 1}
        ]
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        order_str = params['order']
        retain_graph = params['retain_graph']
        
        # Detect which loss inputs are connected
        connected_losses = []
        loss_connections = {}
        
        for i in range(1, 5):  # loss1 through loss4
            loss_name = f"loss{i}"
            if "inputs" in connections and loss_name in connections["inputs"]:
                connected_losses.append(i)
                # Store the connection info for later use
                loss_connections[i] = connections["inputs"][loss_name]
        
        if not connected_losses:
            raise ValueError(f"TrainingSequencer {node_id}: No loss inputs connected")
        
        # Parse and validate order widget
        try:
            order = [int(x.strip()) for x in order_str.split(',') if x.strip()]
        except ValueError:
            raise ValueError(
                f"TrainingSequencer {node_id}: Invalid order '{order_str}'. "
                f"Expected comma-separated integers (e.g., '1,2,3')"
            )
        
        # Validate order contains only connected losses
        for idx in order:
            if idx not in connected_losses:
                raise ValueError(
                    f"TrainingSequencer {node_id}: Order specifies loss{idx} but it's not connected. "
                    f"Connected losses: {', '.join(f'loss{i}' for i in connected_losses)}"
                )
        
        # If order doesn't include all connected losses, add them at the end
        for loss_idx in connected_losses:
            if loss_idx not in order:
                order.append(loss_idx)
        
        # Collect optimizer node IDs from output connections
        optimizer_node_ids = []
        for i in range(1, 5):  # to_opt1 through to_opt4
            output_name = f"to_opt{i}"
            
            # Use follow_node_connection to find optimizer
            opt_node_id = export_utils.follow_node_connection(node_id, output_name)
            
            if opt_node_id and i in connected_losses:
                # Only include if corresponding loss is connected
                optimizer_node_ids.append(str(opt_node_id))
        
        if len(optimizer_node_ids) != len(connected_losses):
            raise ValueError(
                f"TrainingSequencer {node_id}: Number of connected optimizers ({len(optimizer_node_ids)}) "
                f"must match number of connected losses ({len(connected_losses)})"
            )
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "TrainingSequencer",
            "CONNECTED_LOSSES": connected_losses,  # e.g., [1, 2, 3]
            "OPTIMIZER_NODE_IDS": optimizer_node_ids,  # e.g., ['40', '81']
            "ORDER": order,  # e.g., [2, 1, 3]
            "RETAIN_GRAPH": retain_graph
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import asyncio",
            "from typing import Dict, Any, List",
        ]
    
    @classmethod
    def get_input_names(cls):
        # Only connected inputs will be set as required during export
        return ["loss1", "loss2", "loss3", "loss4"]
    
    @classmethod
    def get_output_names(cls):
        return ["to_opt1", "to_opt2", "to_opt3", "to_opt4"]
    
    @classmethod
    def get_subsystem(cls):
        from ..subsystems import SUBSYSTEM_TRAINING
        return SUBSYSTEM_TRAINING