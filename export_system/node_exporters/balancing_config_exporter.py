#!/usr/bin/env python3
"""
Exporter for BalancingConfig node using queue-based template
"""

from ..graph_exporter import ExportableNode

class BalancingConfigExporter(ExportableNode):
    """Exporter for Balancing Config (virtual node)"""
    
    @classmethod
    def get_template_name(cls):
        return "nodes/balancing_config_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        """Prepare template variables for Balancing Config"""
        # Virtual nodes pass configuration to connected nodes
        # The actual configuration is handled during graph export
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "BalancingConfig",
        }
    
    @classmethod
    def get_imports(cls):
        # Virtual nodes don't need imports
        return []
    
    @classmethod
    def is_virtual(cls):
        """Mark this as a virtual node"""
        return True
    
    @classmethod
    def get_input_names(cls):
        return []
    
    @classmethod
    def get_output_names(cls):
        return ["config"]