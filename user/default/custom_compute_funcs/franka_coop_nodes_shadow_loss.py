"""
Franka Cooperative Control Shadow Environment Loss Function (FIXED)

Computes prediction error for dynamic elements only:
- End-effector position (primary)
- Joint angles (secondary)

Static elements like target_pos are not compared as they don't change.
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

def compute(input: torch.Tensor, extra_args=None) -> torch.Tensor:
    """
    Compute prediction error between actual and predicted DYNAMIC elements.
    
    Network learns map: (action(t), obs(t)) -> pred_obs(t+1)
    
    Observation structure (indices):
    - target_pos: [0-2]    (STATIC - don't compare)
    - eef_pos: [3-5]       (DYNAMIC - compare)
    - eef_quat: [6-9]      (STATIC - derived from joints)
    - joint_theta: [10-18] (DYNAMIC - compare)
    - episode_time: [19]   (STATIC - deterministic)
    
    Args:
        input: [obs(t+1), pred_obs(t+1)]
        Shape: [..., 40] where first 20 are actual, last 20 are predicted
        extra_args: Optional additional arguments (unused in this function)
        
    Returns:
        Scalar loss value (weighted combination of eef and joint errors)
    """
        # Extract actual and predicted observations
    obs = input[..., 0:20]           # Actual obs(t+1)
    pred_obs = input[..., 20:40]     # Predicted obs(t+1)
    
    # Extract DYNAMIC elements only
    # End-effector position (most important for control)
    actual_eef = obs[..., 3:6]       # Actual EEF position (x,y,z)
    pred_eef = pred_obs[..., 3:6]    # Predicted EEF position
    
    # Joint angles (secondary importance)
    actual_joints = obs[..., 10:19]   # 9 joint angles
    pred_joints = pred_obs[..., 10:19] # Predicted joint angles
    
    # Compute L2 distances for each component
    eef_error = torch.norm(pred_eef - actual_eef, p=2, dim=-1)
    joint_error = torch.norm(pred_joints - actual_joints, p=2, dim=-1)
    
    # Normalize joint error to similar scale as EEF
    # Joint angles are in radians [-π, π], normalize by π
    # This brings them to roughly [-1, 1] range like EEF positions
    joint_error_normalized = joint_error / 3.14159
    
    # Weighted combination
    # MIX_COEFF weight on end-effector (what we care about for control)
    # (1 - MIX_COEFF) weight on joints (helps learn dynamics)
    MIX_COEFF = 0.8
    total_loss = MIX_COEFF * eef_error + (1 - MIX_COEFF) * joint_error_normalized

    # Return as scalar loss (mean for batch processing)
    return total_loss.mean()