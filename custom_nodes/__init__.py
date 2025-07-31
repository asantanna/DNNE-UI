# This file makes custom_nodes a package so that absolute imports work
# Individual node files are loaded by nodes.py

# Import all node classes for test compatibility
from .mnist_dataset_visnode import MNISTDatasetNode
from .cifar10_dataset_visnode import CIFAR10DatasetNode
from .batch_sampler_visnode import BatchSamplerNode
from .get_batch_visnode import GetBatchNode
from .linear_layer_visnode import LinearLayerNode
from .conv2d_layer_visnode import Conv2DLayerNode
from .activation_visnode import ActivationNode
from .dropout_visnode import DropoutNode
from .batchnorm_visnode import BatchNormNode
from .flatten_visnode import FlattenNode
from .network_visnode import NetworkNode
from .cross_entropy_loss_visnode import CrossEntropyLossNode
from .accuracy_visnode import AccuracyNode
from .sgd_optimizer_visnode import SGDOptimizerNode
from .training_step_visnode import TrainingStepNode
from .epoch_tracker_visnode import EpochTrackerNode
from .tensor_visualizer_visnode import TensorVisualizerNode
from .isaac_gym_envs_visnode import IsaacGymEnvs
from .cartpole_action_visnode import CartpoleActionNode
from .ppo_config_visnode import PPOConfig as PPOConfigNode
from .ppo_agent_visnode import PPOAgent as PPOAgentNode
from .balancing_config_visnode import BalancingConfig as BalancingConfigNode
from .balancing_visnode import BalancingNode
from .or_visnode import ORNode

# Collect all NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import all node modules and collect their mappings
import os
import importlib

current_dir = os.path.dirname(__file__)
for filename in os.listdir(current_dir):
    if filename.endswith('_visnode.py'):
        module_name = filename[:-3]  # Remove .py extension
        module = importlib.import_module(f'.{module_name}', package='custom_nodes')
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)