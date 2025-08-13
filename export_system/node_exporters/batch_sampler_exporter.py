#!/usr/bin/env python3
"""
Exporter for BatchSampler node using queue-based template
"""

from ..graph_exporter import ExportableNode

class BatchSamplerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/batch_sampler_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader - FAIL-FAST: no defaults
        param_specs = [
            {'name': 'batch_size', 'widget_index': 0},
            {'name': 'shuffle', 'widget_index': 1},
            {'name': 'seed', 'widget_index': 2},
            {'name': 'seed_control', 'widget_index': 3}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Validate required parameters are present
        required_params = ['batch_size', 'shuffle', 'seed', 'seed_control']
        missing_params = [p for p in required_params if params.get(p) is None]
        if missing_params:
            raise ValueError(
                f"BatchSampler node {node_id} missing required parameters: {missing_params}. "
                f"The UI must provide all sampler configuration parameters."
            )
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "BatchSamplerNode",
            "BATCH_SIZE": params['batch_size'],
            "SHUFFLE": params['shuffle'],
            "SEED": params['seed'],
            "SEED_CONTROL": f'"{params["seed_control"]}"'  # Add quotes for string value
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "from torch.utils.data import DataLoader",
        ]
    
    @classmethod
    def get_output_names(cls):
        return ["dataloader", "schema"]
    
    @classmethod
    def get_input_names(cls):
        return ["dataset", "schema"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        """BatchSampler passes through dataset schema but wraps data in DataLoader"""
        # Use the universal parameter reader to get widget values
        param_specs = [
            {'name': 'batch_size', 'widget_index': 0},
            {'name': 'shuffle', 'widget_index': 1},
            {'name': 'seed', 'widget_index': 2},
            {'name': 'seed_control', 'widget_index': 3}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Check if we got the required parameters
        if params.get('batch_size') is None:
            raise ValueError(
                f"BatchSampler node missing widget values. "
                f"Could not extract batch_size parameter from node data."
            )
        
        return {
            "outputs": {
                "dataloader": {
                    "type": "dataloader",
                    "batch_size": params['batch_size'],
                    "shuffle": params['shuffle'],
                    "seed": params['seed'],
                    "seed_control": params['seed_control'],
                    "contains_schema": True  # Indicates this contains schema information
                },
                "schema": {
                    "type": "schema",
                    "value": None  # Will be resolved from input
                }
            }
        }
    
    @classmethod
    def _resolve_schema_value(cls, key, parent_schema, node_data, connections, 
                            node_registry, all_nodes, all_links):
        """Pass through the schema from input"""
        if key == "value" and parent_schema.get("type") == "schema":
            # Get the schema from our "schema" input
            input_schema = cls.get_input_schema(node_data, connections, 
                                              node_registry, all_nodes, all_links)
            
            if "schema" in input_schema and input_schema["schema"]:
                return input_schema["schema"]
                
        return None
    

# Registration function
def register_ml_exporters(exporter):
    """Register all ML node exporters"""
    exporter.register_node("MNISTDataset", MNISTDatasetExporter)
    exporter.register_node("CIFAR10Dataset", CIFAR10DatasetExporter)
    exporter.register_node("LinearLayer", LinearLayerExporter)
    exporter.register_node("Loss", LossExporter)
    exporter.register_node("Optimizer", OptimizerExporter)
    exporter.register_node("Display", DisplayExporter)
    exporter.register_node("GetBatch", GetBatchExporter)
    exporter.register_node("SGDOptimizer", SGDOptimizerExporter)
    exporter.register_node("TrainingStep", TrainingStepExporter)
    exporter.register_node("EpochTracker", EpochTrackerExporter)
    exporter.register_node("BatchSampler", BatchSamplerExporter)
    exporter.register_node("CrossEntropyLoss", CrossEntropyLossExporter)
    exporter.register_node("Network", NetworkExporter)
    # Aliases for compatibility
    exporter.register_node("Linear", LinearLayerExporter)