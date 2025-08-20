"""
Franka Cooperative Control Loss Function

Computes distance-based loss from observations for cooperative joint control.
The loss is shared across all three independent neural networks to encourage
coordination without explicit communication.
"""

import torch

#
# These functions are for node configuration
#

# for node
def get_output_type():
    
    # Loss scalar for training
    return "LOSS_SCALAR"

# for exporter
def get_script_output_schema(initial=True, input_schema=None):
    
    # Output is a scalar loss value
    return {
            "outputs": {
                "tensor": {
                    "type": "tensor",
                    "shape": 1,
                    "flattened_size": 1,
                    "dtype": "float32"
                }
            }
        }


#
# This function gets called at runtime
#

def compute(input: torch.Tensor) -> torch.Tensor:
    """
    Compute distance-based loss from observations.
    
    Based on actual franka_dnne.py implementation schema:
    - target_pos: indices [0, 2] - Target position (x,y,z)
    - eef_pos: indices [3, 5] - End-effector position (x,y,z)
    - eef_quat: indices [6, 9] - End-effector quaternion (not used for loss)
    - joint_theta: indices [10, 16] - 7 joint angles in radians (not used for loss)
    - episode_time: index [17] - Episode elapsed time (not used for loss)
    
    Args:
        input: Observation tensor from Isaac Gym simulator
        
    Returns:
        Scalar loss value (L2 distance from end-effector to target)
    """
    # Extract positions from observation tensor
    target_pos = input[..., 0:3]  # Target position (x,y,z)
    eef_pos = input[..., 3:6]     # End-effector position (x,y,z)
    
    # Compute L2 distance (Euclidean distance)
    distance = torch.norm(eef_pos - target_pos, p=2, dim=-1)
    
    # Return as scalar loss (mean for batch processing)
    # Lower distance = better performance = lower loss
    return distance.mean()