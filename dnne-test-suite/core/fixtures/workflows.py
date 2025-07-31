"""
Sample workflows for testing DNNE functionality.

Provides minimal test workflows that can be used across different test scenarios
without requiring large external dependencies.
"""

# Minimal single LinearLayer workflow
MINIMAL_LINEAR_WORKFLOW = {
    "metadata": {
        "dnne-test": True
    },
    "nodes": [
        {
            "id": 1,
            "type": "LinearLayer",
            "pos": [100, 100],
            "size": [270, 154],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": {},
            "outputs": [
                {"name": "output", "type": "TENSOR", "links": []}
            ],
            "properties": {"Node name for S&R": "LinearLayer"},
            "widgets_values": [784, 10, "none", True, "auto"]
        }
    ],
    "links": []
}

# Simple workflow with proper layer connections (from MNIST_Test)
SIMPLE_DATASET_NETWORK = {
    "metadata": {
        "dnne-test": True
    },
    "nodes": [
        {
            "id": "42",
            "type": "LinearLayer",
            "inputs": {},
            "widgets_values": [128, True, "relu", 0.5, "auto"]
        },
        {
            "id": "43",
            "type": "LinearLayer",
            "inputs": {},
            "widgets_values": [128, True, "relu", 0.5, "auto"]
        },
        {
            "id": "46",
            "type": "LinearLayer",
            "inputs": {},
            "widgets_values": [10, True, "none", 0, "auto"]
        },
        {
            "id": "38",
            "type": "BatchSampler",
            "inputs": {},
            "widgets_values": [64, True, 1993, "randomize"]
        },
        {
            "id": "45",
            "type": "TrainingStep",
            "inputs": {},
            "widgets_values": [0]
        },
        {
            "id": "37",
            "type": "MNISTDataset",
            "inputs": {},
            "widgets_values": ["./data", True, True]
        },
        {
            "id": "56",
            "type": "Network",
            "inputs": {},
            "widgets_values": [True, "epoch", "50", False]
        },
        {
            "id": "62",
            "type": "SGDOptimizer",
            "inputs": {},
            "widgets_values": [0.01, 0.9, 0]
        },
        {
            "id": "64",
            "type": "EpochTracker",
            "inputs": {},
            "widgets_values": [10]
        },
        {
            "id": "50",
            "type": "GetBatch",
            "inputs": {},
            "widgets_values": []
        },
        {
            "id": "61",
            "type": "CrossEntropyLoss",
            "inputs": {},
            "widgets_values": []
        }
    ],
    "links": [
        [67, 37, 0, 38, 0, "DATASET"],
        [68, 37, 1, 38, 1, "SCHEMA"],
        [73, 42, 0, 43, 0, "TENSOR"],
        [81, 43, 0, 46, 0, "TENSOR"],
        [92, 38, 0, 50, 0, "DATALOADER"],
        [93, 38, 1, 50, 1, "SCHEMA"],
        [110, 45, 0, 50, 2, "SYNC"],
        [111, 56, 0, 42, 0, "TENSOR"],
        [112, 46, 0, 56, 1, "TENSOR"],
        [115, 50, 0, 56, 0, "TENSOR"],
        [125, 56, 1, 61, 0, "TENSOR"],
        [126, 50, 1, 61, 1, "TENSOR"],
        [128, 61, 0, 45, 0, "TENSOR"],
        [129, 56, 2, 62, 0, "MODEL"],
        [130, 62, 0, 45, 1, "OPTIMIZER"],
        [132, 50, 3, 64, 0, "DICT"],
        [133, 61, 0, 64, 1, "TENSOR"],
        [134, 61, 1, 64, 2, "*"]
    ]
}

# Complete minimal training workflow
MINIMAL_TRAINING_WORKFLOW = {
    "metadata": {
        "dnne-test": True
    },
    "nodes": [
        {
            "id": "1",
            "type": "MNISTDataset",
            "inputs": {},
            "widgets": {
                "batch_size": 8,  # Small batch for testing
                "download": False
            }
        },
        {
            "id": "2", 
            "type": "BatchSampler",
            "inputs": {},
            "widgets": {
                "batch_size": 8,
                "shuffle": True
            }
        },
        {
            "id": "3",
            "type": "GetBatch",
            "inputs": {},
            "widgets": {}
        },
        {
            "id": "4",
            "type": "LinearLayer",
            "inputs": {},
            "widgets": {
                "in_features": 784,
                "out_features": 128,
                "activation": "relu",
                "bias": True,
                "weight_init": "auto"
            }
        },
        {
            "id": "8",
            "type": "LinearLayer",
            "inputs": {},
            "widgets": {
                "in_features": 128,
                "out_features": 10,
                "activation": "none",
                "bias": True,
                "weight_init": "auto"
            }
        },
        {
            "id": "9",
            "type": "Network",
            "inputs": {},
            "widgets": {
                "device": "cpu"
            }
        },
        {
            "id": "5",
            "type": "CrossEntropyLoss",
            "inputs": {},
            "widgets": {
                "reduction": "mean"
            }
        },
        {
            "id": "6",
            "type": "SGDOptimizer", 
            "inputs": {},
            "widgets": {
                "learning_rate": 0.01,
                "momentum": 0.9
            }
        },
        {
            "id": "7",
            "type": "TrainingStep",
            "inputs": {},
            "widgets": {}
        }
    ],
    "links": [
        [1, "1", 0, "2", 0],      # Dataset.dataset -> BatchSampler.dataset
        [2, "1", 1, "2", 1],      # Dataset.schema -> BatchSampler.schema
        [3, "2", 0, "3", 0],      # BatchSampler.dataloader -> GetBatch.dataloader
        [4, "2", 1, "3", 1],      # BatchSampler.schema -> GetBatch.schema
        [5, "3", 0, "9", 0],      # GetBatch.images -> Network.input
        [6, "9", 0, "4", 0],      # Network.layers -> LinearLayer.input
        [7, "4", 0, "8", 0],      # LinearLayer.output -> LinearLayer.input
        [8, "8", 0, "9", 1],      # LinearLayer.output -> Network.to_output
        [9, "9", 1, "5", 0],      # Network.output -> CrossEntropyLoss.predictions
        [10, "3", 1, "5", 1],     # GetBatch.labels -> CrossEntropyLoss.labels
        [11, "5", 0, "7", 0],     # CrossEntropyLoss.loss -> TrainingStep.loss
        [12, "9", 2, "6", 0],     # Network.model -> SGDOptimizer.model
        [13, "6", 0, "7", 1],     # SGDOptimizer.optimizer -> TrainingStep.optimizer
        [14, "7", 0, "3", 2]      # TrainingStep.trigger -> GetBatch.trigger
    ]
}

# Simple robotics workflow (Cartpole-like)
SIMPLE_ROBOTICS_WORKFLOW = {
    "metadata": {
        "dnne-test": True
    },
    "nodes": [
        {
            "id": "1",
            "type": "IsaacGymEnv",
            "inputs": {},
            "widgets": {
                "task": "Cartpole",
                "num_envs": 4,
                "device": "cpu"
            }
        },
        {
            "id": "2",
            "type": "LinearLayer",
            "inputs": {},
            "widgets": {
                "in_features": 4,
                "out_features": 64,
                "activation": "relu",
                "bias": True,
                "weight_init": "auto"
            }
        },
        {
            "id": "4",
            "type": "LinearLayer",
            "inputs": {},
            "widgets": {
                "in_features": 64,
                "out_features": 2,
                "activation": "none",
                "bias": True,
                "weight_init": "auto"
            }
        },
        {
            "id": "5",
            "type": "Network",
            "inputs": {},
            "widgets": {
                "device": "cpu"
            }
        },
        {
            "id": "3", 
            "type": "IsaacGymStep",
            "inputs": {},
            "widgets": {}
        }
    ],
    "links": [
        [1, "1", 0, "5", 0],      # IsaacGymEnv.observations -> Network.input
        [2, "5", 0, "2", 0],      # Network.layers -> LinearLayer.input
        [3, "2", 0, "4", 0],      # LinearLayer.output -> LinearLayer.input
        [4, "4", 0, "5", 1],      # LinearLayer.output -> Network.to_output
        [5, "5", 1, "3", 0],      # Network.output -> IsaacGymStep.actions
        [6, "1", 1, "3", 1]       # IsaacGymEnv.env -> IsaacGymStep.env
    ]
}

# Invalid workflow for error testing
INVALID_WORKFLOW = {
    "metadata": {
        "dnne-test": True
    },
    "nodes": [
        {
            "id": "1",
            "type": "NonExistentNode",
            "inputs": {},
            "widgets": {}
        }
    ],
    "links": [
        [1, "1", 0, "999", 0]  # Invalid connection - node 999 doesn't exist
    ]
}