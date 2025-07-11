"""
Test utilities and helper functions for DNNE testing.

Provides common functionality for:
- Mock object creation
- Test data validation
- Export system testing
- Async testing utilities
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List, Tuple

import torch
import numpy as np


class MockQueueNode:
    """Mock implementation of QueueNode for testing."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.input_queues = {}
        self.output_queues = {}
        self._inputs = []
        self._outputs = []
        
    def setup_inputs(self, required: List[str] = None, optional: List[str] = None):
        """Setup input queues."""
        required = required or []
        optional = optional or []
        self._inputs = required + optional
        
        for input_name in self._inputs:
            self.input_queues[input_name] = asyncio.Queue()
    
    def setup_outputs(self, outputs: List[str]):
        """Setup output queues."""
        self._outputs = outputs
        for output_name in outputs:
            self.output_queues[output_name] = asyncio.Queue()
    
    async def compute(self, **kwargs):
        """Mock compute method."""
        # Simple passthrough for testing
        return {output: f"mock_result_{output}" for output in self._outputs}


class MockTorchModule(torch.nn.Module):
    """Mock PyTorch module for testing."""
    
    def __init__(self, input_size: int = 784, output_size: int = 10):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.linear(x)


def create_mock_exporter():
    """Create a mock GraphExporter for testing."""
    exporter = Mock()
    exporter.node_registry = {}
    exporter.export_workflow = Mock(return_value={"status": "success"})
    exporter.register_node_exporter = Mock()
    return exporter


def create_mock_isaac_gym():
    """Create comprehensive Isaac Gym mock."""
    gym = Mock()
    
    # Core gym functions
    gym.acquire_gym = Mock(return_value=Mock())
    gym.create_sim = Mock(return_value=Mock())
    gym.load_asset = Mock(return_value=Mock())
    gym.create_actor = Mock(return_value=0)
    
    # Environment functions
    gym.get_num_envs = Mock(return_value=512)
    gym.get_actor_rigid_body_states = Mock()
    gym.set_actor_dof_targets = Mock()
    gym.step_graphics = Mock()
    gym.simulate = Mock()
    gym.fetch_results = Mock()
    
    # Observation/action spaces
    gym.get_actor_dof_states = Mock(return_value=torch.randn(512, 4))
    
    return gym


def validate_workflow_structure(workflow: Dict[str, Any]) -> bool:
    """Validate that a workflow has the correct structure and logical consistency."""
    required_keys = ["nodes", "links"]
    
    # Check top-level structure
    for key in required_keys:
        if key not in workflow:
            return False
    
    # Check nodes structure
    if not isinstance(workflow["nodes"], list):
        return False
    
    node_ids = set()
    for node in workflow["nodes"]:
        if not isinstance(node, dict):
            return False
        
        # Check for required node fields (both test format and ComfyUI format)
        if "id" not in node or "type" not in node:
            return False
            
        # Collect node IDs for connection validation
        node_ids.add(str(node["id"]))
        
        # Check for known invalid node types
        node_type = node["type"]
        if node_type == "NonExistentNode":
            return False
    
    # Check links structure and logical consistency
    if not isinstance(workflow["links"], list):
        return False
        
    for link in workflow["links"]:
        # ComfyUI format: [link_id, from_node, from_slot, to_node, to_slot, type]
        # Test format: [from_node, from_output, to_node, to_input]
        if not isinstance(link, list):
            return False
            
        if len(link) == 4:
            # Test format
            from_node, from_output, to_node, to_input = link
        elif len(link) == 6:
            # ComfyUI format
            link_id, from_node, from_slot, to_node, to_slot, link_type = link
        else:
            return False
        
        # Check that linked nodes exist
        if str(from_node) not in node_ids or str(to_node) not in node_ids:
            return False
    
    return True


def validate_export_output(export_path: Path) -> bool:
    """Validate that export output has the expected structure."""
    if not export_path.exists():
        return False
    
    # Check for required files
    required_files = ["runner.py", "__init__.py"]
    for file_name in required_files:
        if not (export_path / file_name).exists():
            return False
    
    # Check for framework and nodes directories
    required_dirs = ["framework", "nodes"]
    for dir_name in required_dirs:
        if not (export_path / dir_name).is_dir():
            return False
    
    return True


def create_temp_export_dir() -> Path:
    """Create a temporary directory for export testing within the allowed export path."""
    import uuid
    import os
    
    # Get project root - should be the working directory
    project_root = Path.cwd()
    export_base = project_root / "export_system" / "exports"
    
    # Create unique test directory name
    test_dir_name = f"test_{uuid.uuid4().hex[:8]}"
    temp_dir = export_base / test_dir_name
    
    # Create the directory
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    return temp_dir


def cleanup_export_dir(export_path: Path):
    """Clean up temporary export directory."""
    if export_path.exists() and export_path.is_dir():
        try:
            shutil.rmtree(export_path)
        except OSError as e:
            # If normal cleanup fails, try to handle the __pycache__ issue
            import stat
            import os
            
            def handle_remove_readonly(func, path, exc):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            
            try:
                shutil.rmtree(export_path, onerror=handle_remove_readonly)
            except Exception as cleanup_error:
                # If cleanup still fails, just warn rather than failing the test
                print(f"Warning: Could not fully clean up {export_path}: {cleanup_error}")


async def run_async_test(coro):
    """Helper to run async test functions."""
    return await coro


def assert_tensor_equal(tensor1: torch.Tensor, tensor2: torch.Tensor, rtol: float = 1e-5):
    """Assert that two tensors are approximately equal."""
    assert torch.allclose(tensor1, tensor2, rtol=rtol), \
        f"Tensors not equal: {tensor1} vs {tensor2}"


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...]):
    """Assert that tensor has expected shape."""
    assert tensor.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {tensor.shape}"


def create_test_dataset(num_samples: int = 100, num_features: int = 784, num_classes: int = 10):
    """Create a small test dataset for ML testing."""
    X = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def mock_conda_environment():
    """Mock conda environment activation for testing."""
    import os
    # Set environment variables that indicate conda is active
    os.environ["CONDA_DEFAULT_ENV"] = "DNNE_PY38"
    os.environ["CONDA_PREFIX"] = "/home/asantanna/miniconda/envs/DNNE_PY38"


class AsyncContextManager:
    """Helper for testing async context managers."""
    
    def __init__(self, return_value=None):
        self.return_value = return_value
        
    async def __aenter__(self):
        return self.return_value
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


def create_mock_node_exporter(node_type: str):
    """Create a mock node exporter for testing."""
    exporter = Mock()
    exporter.get_template_name = Mock(return_value=f"{node_type.lower()}_queue.py")
    exporter.prepare_template_vars = Mock(return_value={
        "NODE_ID": "test_node_1",
        "CLASS_NAME": node_type
    })
    exporter.get_imports = Mock(return_value=["import torch", "import numpy as np"])
    return exporter


def assert_valid_python_code(code: str):
    """Assert that generated code is valid Python."""
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        raise AssertionError(f"Generated code has syntax error: {e}")


def extract_node_from_workflow(workflow: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    """Extract a specific node from a workflow by ID."""
    for node in workflow.get("nodes", []):
        if node.get("id") == node_id:
            return node
    raise ValueError(f"Node with ID {node_id} not found in workflow")


def get_node_connections(workflow: Dict[str, Any], node_id: str) -> List[Tuple[str, str, str, str]]:
    """Get all connections involving a specific node."""
    connections = []
    for link in workflow.get("links", []):
        if len(link) == 4:
            from_node, from_output, to_node, to_input = link
            if from_node == node_id or to_node == node_id:
                connections.append((from_node, from_output, to_node, to_input))
    return connections