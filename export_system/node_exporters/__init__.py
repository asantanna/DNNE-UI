"""
Node exporter classes that handle code generation
"""

# Direct imports - simpler and more reliable
from .ml_nodes import (
    LinearLayerExporter, 
    MNISTDatasetExporter, 
    BatchSamplerExporter, 
    GetBatchExporter,
    CrossEntropyLossExporter, 
    SGDOptimizerExporter,
    AccuracyExporter, 
    TrainingStepExporter
)

# Register all exporters
def register_all_exporters(exporter):
    """Register all node exporters with the graph exporter"""
    # Data nodes
    exporter.register_node("MNISTDataset", MNISTDatasetExporter)
    exporter.register_node("BatchSampler", BatchSamplerExporter) 
    exporter.register_node("GetBatch", GetBatchExporter)
    
    # Layer nodes
    exporter.register_node("LinearLayer", LinearLayerExporter)
    
    # Training nodes
    exporter.register_node("CrossEntropyLoss", CrossEntropyLossExporter)
    exporter.register_node("SGDOptimizer", SGDOptimizerExporter)
    exporter.register_node("TrainingStep", TrainingStepExporter)
    exporter.register_node("Accuracy", AccuracyExporter)
    
    # Add more as implemented