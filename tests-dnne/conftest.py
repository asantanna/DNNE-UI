"""
Pytest configuration and fixtures for DNNE tests.

Provides common fixtures for:
- Sample workflows
- Mock nodes
- Test data
- Environment setup
"""

import os
import sys
import json
import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "fixtures"


# Removed custom event_loop fixture to avoid deprecation warning
# pytest-asyncio will handle event loop creation automatically


@pytest.fixture
def sample_mnist_workflow():
    """Load the MNIST Test workflow for testing."""
    workflow_path = PROJECT_ROOT / "user" / "default" / "workflows" / "MNIST Test.json"
    if workflow_path.exists():
        with open(workflow_path, 'r') as f:
            return json.load(f)
    return None


@pytest.fixture
def sample_cartpole_workflow():
    """Load the Cartpole RL workflow for testing."""
    workflow_path = PROJECT_ROOT / "user" / "default" / "workflows" / "Cartpole_RL_Single.json"
    if workflow_path.exists():
        with open(workflow_path, 'r') as f:
            return json.load(f)
    return None


@pytest.fixture
def minimal_workflow():
    """Create a minimal test workflow with just a few nodes."""
    return {
        "nodes": [
            {
                "id": "1",
                "type": "MNISTDataset", 
                "inputs": {},
                "widgets": {"batch_size": 32}
            },
            {
                "id": "2", 
                "type": "LinearLayer",
                "inputs": {},
                "widgets": {"in_features": 784, "out_features": 10}
            }
        ],
        "links": [
            ["1", "dataset", "2", "input"]
        ]
    }


@pytest.fixture
def temp_export_dir():
    """Create a temporary directory for export testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_torch():
    """Mock PyTorch for tests that don't need actual tensor operations."""
    mock_torch = Mock()
    mock_torch.nn = Mock()
    mock_torch.optim = Mock()
    mock_torch.device = Mock(return_value="cpu")
    mock_torch.cuda = Mock()
    mock_torch.cuda.is_available = Mock(return_value=False)
    return mock_torch


@pytest.fixture
def mock_isaac_gym():
    """Mock Isaac Gym for robotics tests."""
    mock_gym = Mock()
    mock_gym.acquire_gym = Mock()
    mock_gym.create_sim = Mock()
    mock_gym.load_asset = Mock()
    return mock_gym


@pytest.fixture 
def sample_node_data():
    """Sample node data for testing node exporters."""
    return {
        "inputs": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "device": "cuda"
        },
        "widgets": {
            "in_features": 784,
            "out_features": 10,
            "bias": True
        }
    }


@pytest.fixture
def mock_queue_node():
    """Mock QueueNode for testing queue framework."""
    mock_node = AsyncMock()
    mock_node.node_id = "test_node_1"
    mock_node.input_queues = {}
    mock_node.output_queues = {}
    mock_node.setup_inputs = Mock()
    mock_node.setup_outputs = Mock()
    return mock_node


@pytest.fixture
def export_test_registry():
    """Create a test registry for export system testing."""
    from export_system.graph_exporter import GraphExporter
    from export_system.node_exporters import register_all_exporters
    
    exporter = GraphExporter()
    register_all_exporters(exporter)
    return exporter


# Pytest markers for conditional skipping
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "isaac_gym: mark test as requiring Isaac Gym")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle conditional skips."""
    # Skip GPU tests if CUDA not available
    try:
        import torch
        if not torch.cuda.is_available():
            skip_gpu = pytest.mark.skip(reason="CUDA not available")
            for item in items:
                if "gpu" in item.keywords:
                    item.add_marker(skip_gpu)
    except ImportError:
        pass
    
    # Skip Isaac Gym tests if not installed
    try:
        import isaacgym
    except ImportError:
        skip_isaac = pytest.mark.skip(reason="Isaac Gym not installed")
        for item in items:
            if "isaac_gym" in item.keywords:
                item.add_marker(skip_isaac)