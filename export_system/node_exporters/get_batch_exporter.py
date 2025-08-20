#!/usr/bin/env python3
"""
Exporter for GetBatch node using queue-based template
"""

from ..graph_exporter import ExportableNode
from ..subsystems import SUBSYSTEM_DATA

class GetBatchExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/get_batch_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "GetBatchNode"
        }
    
    @classmethod
    def get_imports(cls):
        return []
    
    @classmethod
    def get_output_names(cls):
        return ["images", "labels", "epoch_complete", "epoch_stats"]
    
    @classmethod
    def get_input_names(cls):
        return ["dataloader", "schema", "trigger"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        # GetBatch initial schema - tensor dimensions will be resolved from schema input
        return {
            "outputs": {
                "images": {
                    "type": "tensor",
                    "shape": None,  # To be resolved from schema input
                    "flattened_size": None,  # To be resolved from schema input
                    "dtype": None  # To be resolved from schema input
                },
                "labels": {
                    "type": "tensor",
                    "shape": None,  # To be resolved from schema input
                    "flattened_size": None,  # To be resolved from schema input
                    "dtype": None,  # To be resolved from schema input
                    "num_classes": None  # To be resolved from schema input (if present)
                },
                "epoch_complete": {
                    "type": "boolean",
                    "dtype": "bool"
                },
                "epoch_stats": {
                    "type": "dict",
                    "dtype": "dict"
                }
            }
        }
    
    @classmethod
    def _resolve_schema_value(cls, key, parent_schema, node_data, connections, 
                            node_registry, all_nodes, all_links):
        """Resolve tensor schema values from the dataset schema input"""
        # Get the schema from our "schema" input
        input_schema = cls.get_input_schema(node_data, connections, 
                                          node_registry, all_nodes, all_links)
        
        if "schema" in input_schema and input_schema["schema"]:
            dataset_schema = input_schema["schema"]
            
            # Navigate to the dataset contains section
            if "outputs" in dataset_schema:
                dataset_outputs = dataset_schema["outputs"]
                
                if "dataset" in dataset_outputs and "contains" in dataset_outputs["dataset"]:
                    contains = dataset_outputs["dataset"]["contains"]
                    
                    # Determine if we're resolving for images or labels
                    # We can check if the parent has 'num_classes' to identify labels
                    is_labels = "num_classes" in parent_schema
                    
                    # Get the appropriate source schema
                    if is_labels and "labels" in contains:
                        source_schema = contains["labels"]
                    elif not is_labels and "images" in contains:
                        source_schema = contains["images"]
                    else:
                        return None
                    
                    # Return the requested field from the source schema
                    if key in source_schema:
                        return source_schema[key]
                        
        return None

    @classmethod
    def get_subsystem(cls):
        return SUBSYSTEM_DATA
    
