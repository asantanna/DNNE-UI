# DNNE Test Suite

Comprehensive testing for DNNE-specific features, focusing on ML nodes, robotics integration, export system, and queue-based execution.

## Quick Start

### Install Test Dependencies
```bash
pip install -r tests-dnne/requirements.txt
```

### Run All DNNE Tests
```bash
pytest tests-dnne/
```

## Test Categories

### ML Node Tests
Test machine learning components (LinearLayer, MNISTDataset, Network, etc.):
```bash
pytest tests-dnne/ -m ml
```

### Robotics Tests  
Test Isaac Gym integration and robotics nodes:
```bash
pytest tests-dnne/ -m robotics
```

### Export System Tests
Test workflow-to-code generation:
```bash
pytest tests-dnne/ -m export
```

### Integration Tests
Test complete end-to-end workflows:
```bash
pytest tests-dnne/ -m integration
```

### Performance Tests
Run benchmark and performance tests:
```bash
pytest tests-dnne/ -m performance
```

## Test Structure

```
tests-dnne/
├── unit/                    # Unit tests for individual components
│   ├── ml_nodes/           # ML node tests (LinearLayer, MNISTDataset, etc.)
│   ├── robotics_nodes/     # Robotics node tests (Isaac Gym, sensors)
│   ├── export_system/      # Export system component tests
│   └── queue_framework/    # Queue-based execution tests
├── integration/            # End-to-end workflow tests
├── fixtures/               # Test data and fixtures
├── conftest.py            # Pytest configuration and shared fixtures
├── requirements.txt       # Test dependencies
└── README.md             # This file
```

## Writing Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`

### Test Markers
Use pytest markers to categorize tests:
```python
import pytest

@pytest.mark.ml
def test_linear_layer_forward():
    """Test LinearLayer forward pass."""
    pass

@pytest.mark.robotics
@pytest.mark.isaac_gym
def test_cartpole_environment():
    """Test Cartpole environment initialization."""
    pass

@pytest.mark.export
def test_mnist_workflow_export():
    """Test MNIST workflow export to Python code."""
    pass
```

### Available Fixtures
Common fixtures available in all tests:
- `sample_mnist_workflow` - MNIST Test.json workflow
- `sample_cartpole_workflow` - Cartpole RL workflow  
- `minimal_workflow` - Simple 2-node test workflow
- `temp_export_dir` - Temporary directory for export testing
- `mock_torch` - Mocked PyTorch for lightweight testing
- `mock_isaac_gym` - Mocked Isaac Gym for robotics testing
- `export_test_registry` - Pre-configured export system

### Example Test
```python
import pytest
from custom_nodes.ml_nodes.layer_nodes import LinearLayer

@pytest.mark.ml
def test_linear_layer_creation(sample_node_data):
    \"\"\"Test LinearLayer node creation and parameter extraction.\"\"\"
    node = LinearLayer()
    
    # Test node properties
    assert hasattr(node, 'INPUT_TYPES')
    assert hasattr(node, 'RETURN_TYPES')
    
    # Test parameter processing
    inputs = sample_node_data['inputs']
    result = node.execute(**inputs)
    assert result is not None
```

## Environment Setup

### Conda Environment
Ensure you're in the correct conda environment:
```bash
source /home/asantanna/miniconda/bin/activate DNNE_PY38
```

### Isaac Gym (Optional)
For robotics tests requiring Isaac Gym:
1. Ensure Isaac Gym is installed at `~/isaacgym`
2. Set `ISAAC_GYM_ROOT_DIR` environment variable if needed
3. Tests will be skipped automatically if Isaac Gym is not available

## Test Execution Options

### Parallel Execution
Run tests in parallel for faster execution:
```bash
pytest tests-dnne/ -n auto
```

### Verbose Output
Get detailed test output:
```bash
pytest tests-dnne/ -v
```

### Coverage Reporting
Generate test coverage reports:
```bash
pytest tests-dnne/ --cov=custom_nodes --cov=export_system
```

### Skip Slow Tests
Skip time-consuming tests:
```bash
pytest tests-dnne/ -m "not slow"
```

### Skip GPU Tests
Skip tests requiring CUDA:
```bash
pytest tests-dnne/ -m "not gpu"
```

## Debugging Tests

### Run Single Test
```bash
pytest tests-dnne/unit/ml_nodes/test_linear_layer.py::test_linear_layer_creation -v
```

### Debug Mode
Drop into debugger on failures:
```bash
pytest tests-dnne/ --pdb
```

### Print Statements
Enable print statements in tests:
```bash
pytest tests-dnne/ -s
```

## Continuous Integration

Tests are designed to run in CI environments:
- Automatic skipping of GPU/Isaac Gym tests when not available
- Lightweight mocking for external dependencies
- Fast unit tests separate from slow integration tests

## Troubleshooting

### Common Issues

**Import Errors**
- Ensure you're in the correct conda environment
- Check that project root is in Python path
- Verify all dependencies are installed

**Missing Test Data**
- Check that workflow files exist in `user/default/workflows/`
- Ensure test fixtures are properly created

**Isaac Gym Tests Failing**
- Verify Isaac Gym installation at `~/isaacgym`
- Check that `import isaacgym` works in Python
- Tests will skip automatically if Isaac Gym is not available

**CUDA Tests Failing**
- Tests will skip automatically if CUDA is not available
- Use `-m "not gpu"` to explicitly skip GPU tests

### Getting Help

1. Check test output for specific error messages
2. Run tests with `-v` flag for detailed information
3. Use `--tb=long` for full tracebacks
4. Check the specific test file for documentation

## Contributing

When adding new DNNE features:
1. Add corresponding unit tests in appropriate category
2. Use proper test markers (`@pytest.mark.ml`, etc.)
3. Include integration tests for end-to-end functionality
4. Update this README if adding new test categories
5. Ensure tests can run without external dependencies (use mocking)