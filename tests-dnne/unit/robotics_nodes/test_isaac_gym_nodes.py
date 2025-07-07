"""
Unit tests for Isaac Gym robotics nodes.

Tests Isaac Gym environment and step nodes with proper import order handling.
"""

import pytest
import sys
from unittest.mock import Mock, patch

# Isaac Gym must be imported - no skipping
import isaacgym
from custom_nodes.robotics_nodes.isaac_gym_nodes import IsaacGymEnvNode, IsaacGymStepNode, ISAAC_GYM_AVAILABLE

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
    """Test IsaacGymEnvNode basic structure."""
    node = IsaacGymEnvNode()
    
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
    
    # Should have simulation handle output
    assert "SIM_HANDLE" in return_types or any("sim" in name.lower() for name in return_names)


@pytest.mark.robotics
@pytest.mark.isaac_gym  
@pytest.mark.timeout(30)
def test_isaac_gym_step_node_structure():
    """Test IsaacGymStepNode basic structure."""
    node = IsaacGymStepNode()
    
    # Test basic node structure
    assert hasattr(node, 'INPUT_TYPES')
    assert hasattr(node, 'RETURN_TYPES') 
    assert hasattr(node, 'RETURN_NAMES')
    
    # Check input types
    input_types = node.INPUT_TYPES()
    assert isinstance(input_types, dict)
    assert "required" in input_types
    
    # Should accept actions and sim handle
    required = input_types["required"]
    optional = input_types.get("optional", {})
    all_inputs = {**required, **optional}
    
    # Should have action input
    has_actions = any("action" in key.lower() for key in all_inputs.keys())
    assert has_actions, f"Should have action input. Available: {list(all_inputs.keys())}"
    
    # Check return types
    return_types = node.RETURN_TYPES
    return_names = node.RETURN_NAMES
    assert len(return_types) == len(return_names)
    
    # Should return observations, rewards, done
    has_observations = "TENSOR" in return_types or any("obs" in name.lower() for name in return_names)
    assert has_observations, "Should return observations"


@pytest.mark.robotics
@pytest.mark.timeout(30)
def test_or_node_structure():
    """Test OR node basic structure."""
    from custom_nodes.robotics_nodes.isaac_gym_nodes import ORNode
    
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
    from custom_nodes.robotics_nodes.isaac_gym_nodes import ORNode
    
    node = ORNode()
    assert hasattr(node, "CATEGORY")
    
    category = node.CATEGORY.lower()
    assert any(keyword in category for keyword in ["robotics", "control", "rl", "dnne"])


@pytest.mark.robotics
@pytest.mark.timeout(30)
def test_isaac_gym_import_order_awareness():
    """Test that Isaac Gym nodes are aware of import order issues."""
    # Since we import Isaac Gym at the top, it should be available
    assert isinstance(ISAAC_GYM_AVAILABLE, bool), "ISAAC_GYM_AVAILABLE should be a boolean"
    
    # In our strict test environment, Isaac Gym must be available
    assert ISAAC_GYM_AVAILABLE, "Isaac Gym must be available in test environment"