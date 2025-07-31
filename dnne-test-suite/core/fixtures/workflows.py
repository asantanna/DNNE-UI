"""
Sample workflows for testing DNNE functionality.

Provides minimal test workflows that can be used across different test scenarios
without requiring large external dependencies.
"""

# Minimal single LinearLayer workflow
MINIMAL_LINEAR_WORKFLOW = {
    "nodes": [
        {
            "id": "1",
            "type": "LinearLayer",
            "inputs": {},
            "widgets": {
                "in_features": 784,
                "out_features": 10,
                "bias": True
            }
        }
    ],
    "links": []
}

# Simple workflow with proper layer connections
SIMPLE_DATASET_NETWORK = {
    "nodes": [
        {
            "id": "1", 
            "type": "MNISTDataset",
            "inputs": {},
            "widgets": {
                "batch_size": 32,
                "download": False
            }
        },
        {
            "id": "2",
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
            "id": "3",
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
            "id": "4",
            "type": "Network", 
            "inputs": {},
            "widgets": {
                "device": "cpu"
            }
        }
    ],
    "links": [
        ["1", "dataset", "4", "input"],      # Dataset connects to network
        ["4", "to_layers", "2", "input"],    # Network's to_layers connects to first layer
        ["2", "output", "3", "input"],       # First layer to second layer
        ["3", "output", "4", "to_output"]    # Second layer back to network's to_output
    ]
}

# Complete minimal training workflow
MINIMAL_TRAINING_WORKFLOW = {
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
        ["1", "dataset", "2", "dataset"],
        ["2", "sampler", "3", "sampler"],
        ["3", "batch", "9", "input"],
        ["9", "to_layers", "4", "input"],    # Network's to_layers connects to first layer
        ["4", "output", "8", "input"],
        ["8", "output", "9", "to_output"],
        ["9", "predictions", "5", "predictions"],
        ["3", "targets", "5", "targets"],
        ["5", "loss", "7", "loss"],
        ["9", "model", "6", "model"],
        ["6", "optimizer", "7", "optimizer"],
        ["7", "ready_signal", "3", "trigger"]
    ]
}

# Simple robotics workflow (Cartpole-like)
SIMPLE_ROBOTICS_WORKFLOW = {
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
        ["1", "observations", "5", "input"],  # Observations to network
        ["5", "to_layers", "2", "input"],     # Network's to_layers to first layer
        ["2", "output", "4", "input"],        # First layer to second layer
        ["4", "output", "5", "to_output"],    # Second layer back to network
        ["5", "actions", "3", "actions"],     # Network actions to step
        ["1", "env", "3", "env"]              # Environment to step
    ]
}

# Invalid workflow for error testing
INVALID_WORKFLOW = {
    "nodes": [
        {
            "id": "1",
            "type": "NonExistentNode",
            "inputs": {},
            "widgets": {}
        }
    ],
    "links": [
        ["1", "output", "999", "input"]  # Invalid connection
    ]
}