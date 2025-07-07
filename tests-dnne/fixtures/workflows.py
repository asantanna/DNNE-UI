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

# Simple 2-node workflow: Dataset -> Network
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
            "type": "Network", 
            "inputs": {},
            "widgets": {
                "device": "cpu"
            }
        }
    ],
    "links": [
        ["1", "dataset", "2", "input"]
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
        ["3", "batch", "4", "input"], 
        ["4", "predictions", "5", "predictions"],
        ["3", "targets", "5", "targets"],
        ["5", "loss", "7", "loss"],
        ["4", "model", "6", "model"],
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
        ["1", "observations", "2", "input"],
        ["2", "actions", "3", "actions"],
        ["1", "env", "3", "env"]
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