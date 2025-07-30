#!/usr/bin/env python3
"""
Exporter for IsaacGymEnvs node using queue-based template
"""

from ..graph_exporter import ExportableNode

class IsaacGymEnvsExporter(ExportableNode):
    """Exporter for IsaacGymEnvs virtual node"""
    
    @classmethod
    def is_virtual(cls):
        """IsaacGymEnvs is a virtual node - only provides configuration"""
        return True
    
    @classmethod
    def get_template_name(cls):
        # Virtual nodes don't need templates
        return None
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Virtual nodes don't generate code
        return {}
    
    @classmethod
    def get_imports(cls):
        return []
    
    @classmethod
    def get_output_names(cls):
        return ["env"]
    
    @classmethod
    def get_input_names(cls):
        return []


# Registration function