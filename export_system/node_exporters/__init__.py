"""
Node exporter classes that handle code generation
"""

from .ml_nodes import *
from .robotics_nodes import *

# Register all exporters
def register_all_exporters(exporter):
    """Register all node exporters with the graph exporter"""
    # ML nodes
    exporter.register_node("LinearLayer", LinearLayerExporter)
    exporter.register_node("MNISTDataset", MNISTDatasetExporter)
    exporter.register_node("BatchSampler", BatchSamplerExporter)
    # Add more as implemented
