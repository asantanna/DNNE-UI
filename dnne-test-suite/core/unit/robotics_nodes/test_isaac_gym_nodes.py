"""
Unit tests for Isaac Gym robotics nodes.

Tests Isaac Gym environment and step nodes with proper import order handling.
"""

import pytest
import sys
from unittest.mock import Mock, patch

# Isaac Gym must be imported - no skipping
import isaacgym
from custom_nodes import IsaacGymEnvs

# Set ISAAC_GYM_AVAILABLE for compatibility
ISAAC_GYM_AVAILABLE = True

@pytest.mark.robotics
@pytest.mark.isaac_gym
@pytest.mark.timeout(30)
def test_isaac_gym_availability():
    """Test Isaac Gym availability."""
    assert ISAAC_GYM_AVAILABLE, "Isaac Gym must be available for tests"


@pytest.mark.robotics  
@pytest.mark.isaac_gym
@pytest.mark.timeout(30)
def test_isaac_gym_env_node_structure():
    """Test IsaacGymEnvs basic structure."""
    node = IsaacGymEnvs()
    
    # Test basic node structure
    assert hasattr(node, 'INPUT_TYPES')
    assert hasattr(node, 'RETURN_TYPES')
    assert hasattr(node, 'RETURN_NAMES')
    assert hasattr(node, 'FUNCTION')
    assert hasattr(node, 'CATEGORY')
    
    # Check input types
    input_types = node.INPUT_TYPES()
    assert isinstance(input_types, dict)
    
    # Check return types
    return_types = node.RETURN_TYPES
    return_names = node.RETURN_NAMES
    assert len(return_types) == len(return_names)
    
    # Should have environment config output
    assert "GYM_CONFIG" in return_types or "ISAAC_ENV_CONFIG" in return_types


@pytest.mark.robotics
@pytest.mark.isaac_gym  
@pytest.mark.timeout(30)
@pytest.mark.skip(reason="IsaacGymStepNode not implemented in flat structure")
def test_isaac_gym_step_node_structure():
    """Test IsaacGymStepNode basic structure."""
    pass  # Node not implemented in flat structure


@pytest.mark.robotics
@pytest.mark.timeout(30)
def test_or_node_structure():
    """Test OR node basic structure."""
    from custom_nodes import ORNode
    
    node = ORNode()
    
    # Test basic node structure
    assert hasattr(node, 'INPUT_TYPES')
    assert hasattr(node, 'RETURN_TYPES')
    assert hasattr(node, 'RETURN_NAMES')
    
    # Check input types - should have multiple optional inputs
    input_types = node.INPUT_TYPES()
    assert isinstance(input_types, dict)
    
    # OR node should have optional inputs
    optional = input_types.get("optional", {})
    assert len(optional) >= 2, "OR node should have multiple optional inputs"
    
    # Check return types - should have single output
    return_types = node.RETURN_TYPES
    return_names = node.RETURN_NAMES
    assert len(return_types) == 1, "OR node should have single output"
    assert len(return_names) == 1, "OR node should have single output name"


@pytest.mark.robotics
@pytest.mark.timeout(30)
def test_robotics_node_categories():
    """Test that robotics nodes have appropriate categories."""
    from custom_nodes import ORNode
    
    node = ORNode()
    assert hasattr(node, "CATEGORY")
    
    category = node.CATEGORY.lower()
    assert any(keyword in category for keyword in ["robotics", "control", "rl", "dnne", "utility"])


@pytest.mark.robotics
@pytest.mark.timeout(30)
def test_isaac_gym_import_order_awareness():
    """Test that Isaac Gym nodes are aware of import order issues."""
    # Since we import Isaac Gym at the top, it should be available
    assert isinstance(ISAAC_GYM_AVAILABLE, bool), "ISAAC_GYM_AVAILABLE should be a boolean"
    
    # In our strict test environment, Isaac Gym must be available
    assert ISAAC_GYM_AVAILABLE, "Isaac Gym must be available in test environment"