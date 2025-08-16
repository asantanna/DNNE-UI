#!/usr/bin/env python3
"""
Test script to verify that node color defaults are working
"""

import sys
sys.path.append('.')

# Import some nodes to check their color properties
from custom_nodes.ml_nodes.data_nodes import MNISTDatasetNode, BatchSamplerNode, GetBatchNode, CIFAR10DatasetNode
from custom_nodes.ml_nodes.training_nodes import CrossEntropyLossNode, SGDOptimizerNode, TrainingStepNode, EpochTrackerNode
from custom_nodes.ml_nodes.layer_nodes import LinearLayerNode, NetworkNode
from custom_nodes.utility_nodes.balancing_node import BalancerNode
from custom_nodes.utility_nodes.balancing_config import BalancerConfig
from custom_nodes.rl_nodes.ppo_agent import PPOAgent
from custom_nodes.rl_nodes.ppo_config import PPOConfig

# Test nodes
test_nodes = [
    ("Data Nodes", [MNISTDatasetNode, CIFAR10DatasetNode, BatchSamplerNode, GetBatchNode]),
    ("Training Nodes", [CrossEntropyLossNode, SGDOptimizerNode, TrainingStepNode, EpochTrackerNode]),
    ("Layer Nodes", [LinearLayerNode]),
    ("Network Nodes", [NetworkNode]),
    ("Balancer Nodes", [BalancerNode, BalancerConfig]),
    ("RL Nodes", [PPOAgent, PPOConfig])
]

print("Node Color Test Results:")
print("=" * 60)

for category, nodes in test_nodes:
    print(f"\n{category}:")
    for node_class in nodes:
        color = getattr(node_class, 'COLOR', 'NOT SET')
        bgcolor = getattr(node_class, 'BGCOLOR', 'NOT SET')
        print(f"  {node_class.__name__:<25} COLOR={color:<10} BGCOLOR={bgcolor}")

print("\n" + "=" * 60)
print("Expected colors:")
print("  Data nodes:      COLOR=#332922    BGCOLOR=#593930  (Brown)")
print("  Training nodes:  COLOR=#432       BGCOLOR=#653      (Yellow)")
print("  Layer nodes:     COLOR=#232       BGCOLOR=#353      (Green)")
print("  Network nodes:   COLOR=#223       BGCOLOR=#335      (Blue)")
print("  Balancer nodes: COLOR=#323       BGCOLOR=#535      (Purple)")
print("  RL nodes:        COLOR=#322       BGCOLOR=#533      (Red)")