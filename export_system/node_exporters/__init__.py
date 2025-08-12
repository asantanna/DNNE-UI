"""
Node exporter classes that handle code generation using queue-based templates
Updated to use flat structure with individual exporter files
"""

# ML Exporters
from .mnist_dataset_exporter import MNISTDatasetExporter
from .cifar10_dataset_exporter import CIFAR10DatasetExporter
from .batch_sampler_exporter import BatchSamplerExporter
from .get_batch_exporter import GetBatchExporter
from .linear_layer_exporter import LinearLayerExporter
from .conv2d_layer_exporter import Conv2DLayerExporter
from .activation_exporter import ActivationExporter
from .dropout_exporter import DropoutExporter
from .batchnorm_exporter import BatchNormExporter
from .flatten_exporter import FlattenExporter
from .network_exporter import NetworkExporter
from .cross_entropy_loss_exporter import CrossEntropyLossExporter
from .accuracy_exporter import AccuracyExporter
from .sgd_optimizer_exporter import SGDOptimizerExporter
from .training_step_exporter import TrainingStepExporter
from .epoch_tracker_exporter import EpochTrackerExporter
from .tensor_visualizer_exporter import TensorVisualizerExporter
from .loss_exporter import LossExporter
from .optimizer_exporter import OptimizerExporter
from .display_exporter import DisplayExporter

# Robotics Exporters
from .isaac_gym_envs_exporter import IsaacGymEnvsExporter
from .isaac_gym_sim_exporter import IsaacGymSimExporter
from .camera_sensor_exporter import CameraSensorExporter
from .imu_sensor_exporter import IMUSensorExporter
from .vision_network_exporter import VisionNetworkExporter
from .sound_network_exporter import SoundNetworkExporter
from .decision_network_exporter import DecisionNetworkExporter
from .robot_controller_exporter import RobotControllerExporter

# RL Exporters
from .ppo_config_exporter import PPOConfigExporter
from .ppo_agent_exporter import PPOAgentExporter

# Utility Exporters
from .or_node_exporter import ORNodeExporter
from .balancing_node_exporter import BalancingNodeExporter
from .balancing_config_exporter import BalancingConfigExporter
from .data_streamer_exporter import DataStreamerExporter


# Registration functions
def register_ml_exporters(exporter):
    """Register all ML node exporters"""
    exporter.register_node("MNISTDataset", MNISTDatasetExporter)
    exporter.register_node("CIFAR10Dataset", CIFAR10DatasetExporter)
    exporter.register_node("BatchSampler", BatchSamplerExporter)
    exporter.register_node("GetBatch", GetBatchExporter)
    exporter.register_node("LinearLayer", LinearLayerExporter)
    exporter.register_node("Conv2DLayer", Conv2DLayerExporter)
    exporter.register_node("Activation", ActivationExporter)
    exporter.register_node("Dropout", DropoutExporter)
    exporter.register_node("BatchNorm", BatchNormExporter)
    exporter.register_node("Flatten", FlattenExporter)
    exporter.register_node("Network", NetworkExporter)
    exporter.register_node("CrossEntropyLoss", CrossEntropyLossExporter)
    exporter.register_node("Accuracy", AccuracyExporter)
    exporter.register_node("SGDOptimizer", SGDOptimizerExporter)
    exporter.register_node("TrainingStep", TrainingStepExporter)
    exporter.register_node("EpochTracker", EpochTrackerExporter)
    exporter.register_node("TensorVisualizer", TensorVisualizerExporter)
    exporter.register_node("Loss", LossExporter)
    exporter.register_node("Optimizer", OptimizerExporter)
    exporter.register_node("Display", DisplayExporter)
    
    # Aliases for compatibility
    exporter.register_node("Linear", LinearLayerExporter)

def register_robotics_exporters(exporter):
    """Register all robotics node exporters"""
    exporter.register_node("IsaacGymEnvs", IsaacGymEnvsExporter)
    exporter.register_node("IsaacGymSim", IsaacGymSimExporter)
    exporter.register_node("CameraSensor", CameraSensorExporter)
    exporter.register_node("IMUSensor", IMUSensorExporter)
    exporter.register_node("VisionNetwork", VisionNetworkExporter)
    exporter.register_node("SoundNetwork", SoundNetworkExporter)
    exporter.register_node("DecisionNetwork", DecisionNetworkExporter)
    exporter.register_node("RobotController", RobotControllerExporter)

def register_rl_exporters(exporter):
    """Register all RL node exporters"""
    exporter.register_node("PPOConfig", PPOConfigExporter)
    exporter.register_node("PPOAgent", PPOAgentExporter)

def register_utility_exporters(exporter):
    """Register all utility node exporters"""
    exporter.register_node("ORNode", ORNodeExporter)
    exporter.register_node("BalancingNode", BalancingNodeExporter)
    exporter.register_node("BalancingConfig", BalancingConfigExporter)
    exporter.register_node("DataStreamer", DataStreamerExporter)

# Main registration function
def register_all_exporters(exporter):
    """Register all node exporters with the graph exporter"""
    register_ml_exporters(exporter)
    register_robotics_exporters(exporter)
    register_rl_exporters(exporter)
    register_utility_exporters(exporter)
    
    # Log registration summary
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Registered {len(exporter.node_registry)} node types for export")

# Export all classes for direct access
__all__ = [
    # ML nodes
    'MNISTDatasetExporter',
    'CIFAR10DatasetExporter',
    'BatchSamplerExporter',
    'GetBatchExporter',
    'LinearLayerExporter',
    'Conv2DLayerExporter',
    'ActivationExporter',
    'DropoutExporter',
    'BatchNormExporter',
    'FlattenExporter',
    'NetworkExporter',
    'CrossEntropyLossExporter',
    'AccuracyExporter',
    'SGDOptimizerExporter',
    'TrainingStepExporter',
    'EpochTrackerExporter',
    'TensorVisualizerExporter',
    'LossExporter',
    'OptimizerExporter',
    'DisplayExporter',
    # Robotics nodes
    'IsaacGymEnvsExporter',
    'IsaacGymSimExporter',
    'CameraSensorExporter',
    'IMUSensorExporter',
    'VisionNetworkExporter',
    'SoundNetworkExporter',
    'DecisionNetworkExporter',
    'RobotControllerExporter',
    # RL nodes
    'PPOConfigExporter',
    'PPOAgentExporter',
    # Utility nodes
    'ORNodeExporter',
    'BalancingNodeExporter',
    'BalancingConfigExporter',
    'DataStreamerExporter',
    # Registration functions
    'register_all_exporters',
    'register_ml_exporters',
    'register_robotics_exporters',
    'register_rl_exporters',
    'register_utility_exporters'
]