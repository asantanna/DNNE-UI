"""
Node exporter classes that handle code generation using queue-based templates
"""

from .ml_nodes import (
    MNISTDatasetExporter,
    LinearLayerExporter,
    LossExporter,
    OptimizerExporter,
    DisplayExporter,
    register_ml_exporters
)

from .robotics_nodes import (
    CameraSensorExporter,
    IMUSensorExporter,
    VisionNetworkExporter,
    SoundNetworkExporter,
    DecisionNetworkExporter,
    RobotControllerExporter,
    IsaacGymEnvExporter,
    register_robotics_exporters
)

from .rl_nodes import (
    PPOAgentExporter,
    PPOTrainerExporter,
    register_rl_exporters
)

# Register all exporters
def register_all_exporters(exporter):
    """Register all node exporters with the graph exporter"""
    register_ml_exporters(exporter)
    register_robotics_exporters(exporter)
    register_rl_exporters(exporter)
    
    # Log registration summary
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Registered {len(exporter.node_registry)} node types for export")

# Export all classes for direct access
__all__ = [
    # ML nodes
    'MNISTDatasetExporter',
    'LinearLayerExporter', 
    'LossExporter',
    'OptimizerExporter',
    'DisplayExporter',
    # Robotics nodes
    'CameraSensorExporter',
    'IMUSensorExporter',
    'VisionNetworkExporter',
    'SoundNetworkExporter',
    'DecisionNetworkExporter',
    'RobotControllerExporter',
    'IsaacGymEnvExporter',
    # RL nodes
    'PPOAgentExporter',
    'PPOTrainerExporter',
    # Registration functions
    'register_all_exporters',
    'register_ml_exporters',
    'register_robotics_exporters',
    'register_rl_exporters'
]
