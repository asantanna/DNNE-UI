# Tensor Dimension Standards

## Core Convention (STRICT)

DNNE enforces a strict tensor dimension convention across all nodes:

| Dimension | Purpose | Description | Example |
|-----------|---------|-------------|---------|
| **Dim 0** | Batch/Environment | Number of samples or environments | `[32, ...]` for batch size 32 |
| **Dim 1** | Features | Feature dimension for concatenation | `[32, 128]` for 128 features |
| **Dim 2+** | Data Dimensions | Additional data (H/W for images, L for sequences) | `[32, 3, 224, 224]` for images |

## Key Principles

### 1. **Never Use 1D Tensors for Data**
1D tensors are only allowed for scalar losses/rewards. All data tensors must be at least 2D.

```python
# ❌ WRONG - 1D tensor
features = torch.randn(128)  # Shape: [128]

# ✅ CORRECT - 2D tensor with batch dimension
features = torch.randn(1, 128)  # Shape: [1, 128]

# ✅ CORRECT - Auto-fix for 1D inputs
if tensor.dim() == 1:
    tensor = tensor.unsqueeze(0)  # Add batch dimension
```

### 2. **Fail-Fast on Dimension Mismatches**
Nodes must raise `ValueError` if dimensions don't match convention—no automatic reshaping that might hide bugs.

```python
def validate_tensor(tensor: torch.Tensor, expected_dims: int, node_id: str):
    """Validate tensor dimensions match DNNE standards"""
    if tensor.dim() != expected_dims:
        raise ValueError(
            f"Node {node_id}: Expected {expected_dims}D tensor, got {tensor.dim()}D. "
            f"Shape: {tensor.shape}. Tensors must follow DNNE dimension standards."
        )
```

### 3. **Concatenation Always on Dim 1**
Concat and split operations ONLY work on dimension 1 (features), never dimension 0 (batch).

```python
# ✅ CORRECT - Concatenate features
combined = torch.cat([tensor_a, tensor_b], dim=1)  # [batch, features_a + features_b]

# ❌ WRONG - Never concatenate batches
combined = torch.cat([tensor_a, tensor_b], dim=0)  # This violates convention!
```

## Common Tensor Patterns

### ML Training Tensors
```python
# Input data
images = torch.randn(32, 3, 224, 224)  # [batch, channels, height, width]
labels = torch.randn(32, 10)           # [batch, num_classes]

# Network layers
hidden = torch.randn(32, 256)          # [batch, hidden_size]
output = torch.randn(32, 10)           # [batch, output_size]

# Loss (scalar per batch, then reduced)
loss = torch.randn(32)                 # [batch] - per-sample loss
loss = loss.mean()                     # scalar - reduced loss
```

### RL Environment Tensors
```python
# Observations
obs = torch.randn(16, 84)              # [num_envs, observation_size]

# Actions
actions = torch.randn(16, 7)           # [num_envs, action_size]

# Rewards (special case - can be 1D)
rewards = torch.randn(16)              # [num_envs] - scalar per env

# Values/advantages
values = torch.randn(16, 1)            # [num_envs, 1] - keep 2D for consistency
```

### Robotics Control Tensors
```python
# Joint positions
joint_pos = torch.randn(4, 7)          # [num_robots, num_joints]

# End-effector pose
ee_pose = torch.randn(4, 6)            # [num_robots, pose_dims]

# Force/torque
forces = torch.randn(4, 3)             # [num_robots, force_components]
torques = torch.randn(4, 3)            # [num_robots, torque_components]
```

## Node-Specific Conventions

### ConcatNode
Concatenates multiple inputs along feature dimension:
```python
class ConcatNode:
    def compute(self, inputs: Dict[str, torch.Tensor]):
        # Validate all inputs are 2D or higher
        for name, tensor in inputs.items():
            if tensor.dim() < 2:
                # Auto-fix 1D tensors
                tensor = tensor.unsqueeze(0)
            
        # Always concatenate on dim=1
        return torch.cat(tensors, dim=1)
```

### SplitNode
Splits tensor along feature dimension:
```python
class SplitNode:
    def compute(self, input: torch.Tensor):
        # Validate input is at least 2D
        if input.dim() < 2:
            raise ValueError(f"SplitNode requires 2D+ input, got {input.dim()}D")
        
        # Split along features (dim=1)
        splits = torch.split(input, self.split_sizes, dim=1)
        return splits
```

### NetworkNode
Processes batched inputs through layers:
```python
class NetworkNode:
    def forward(self, x: torch.Tensor):
        # Expect [batch, features] input
        if x.dim() != 2:
            raise ValueError(f"Network expects 2D input [batch, features], got {x.shape}")
        
        # Process through layers maintaining batch dimension
        for layer in self.layers:
            x = layer(x)  # Preserves dim 0
        return x
```

## Debugging Dimension Issues

### Common Errors and Fixes

#### Error: "Expected 2D tensor, got 1D"
```python
# Problem: Scalar or 1D data
tensor = torch.tensor([1.0, 2.0, 3.0])  # Shape: [3]

# Fix: Add batch dimension
tensor = tensor.unsqueeze(0)  # Shape: [1, 3]
```

#### Error: "Cannot concatenate tensors with different batch sizes"
```python
# Problem: Mismatched dim 0
tensor_a = torch.randn(32, 128)  # Batch size 32
tensor_b = torch.randn(16, 128)  # Batch size 16

# Fix: Ensure same batch size or handle separately
# Option 1: Pad smaller batch
tensor_b = F.pad(tensor_b, (0, 0, 0, 16))  # Pad to [32, 128]

# Option 2: Process in separate batches
```

#### Error: "Dimension mismatch in matrix multiplication"
```python
# Problem: Wrong feature dimensions
input = torch.randn(32, 100)
weight = torch.randn(128, 200)  # Incompatible!

# Fix: Ensure dimensions align
weight = torch.randn(128, 100)  # Now compatible
output = input @ weight.T  # [32, 100] @ [100, 128] = [32, 128]
```

### Debugging Tools

```python
def debug_tensor_shape(tensor: torch.Tensor, name: str):
    """Print detailed tensor shape information"""
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dimensions: {tensor.dim()}")
    print(f"  Batch size (dim 0): {tensor.shape[0] if tensor.dim() > 0 else 'N/A'}")
    print(f"  Feature size (dim 1): {tensor.shape[1] if tensor.dim() > 1 else 'N/A'}")
    print(f"  Additional dims: {tensor.shape[2:] if tensor.dim() > 2 else 'None'}")
```

## Validation Utilities

### Standard Tensor Validator
```python
class TensorValidator:
    @staticmethod
    def validate_batch_tensor(tensor: torch.Tensor, node_id: str, min_dims: int = 2):
        """Validate tensor follows DNNE batch/feature convention"""
        if tensor.dim() < min_dims:
            raise ValueError(
                f"Node {node_id}: Tensor must have at least {min_dims} dimensions "
                f"[batch, features, ...]. Got {tensor.dim()}D tensor with shape {tensor.shape}"
            )
        return True
    
    @staticmethod
    def validate_concat_inputs(tensors: List[torch.Tensor], node_id: str):
        """Validate tensors can be concatenated"""
        if not tensors:
            raise ValueError(f"Node {node_id}: No tensors to concatenate")
        
        # Check all have same batch size
        batch_sizes = [t.shape[0] for t in tensors]
        if len(set(batch_sizes)) > 1:
            raise ValueError(
                f"Node {node_id}: Cannot concat tensors with different batch sizes: {batch_sizes}"
            )
        
        # Check all have same dimensions
        dims = [t.dim() for t in tensors]
        if len(set(dims)) > 1:
            raise ValueError(
                f"Node {node_id}: Cannot concat tensors with different dimensions: {dims}"
            )
        
        return True
```

### Auto-fixing Helper
```python
def ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is at least 2D following DNNE convention"""
    if tensor.dim() == 0:
        # Scalar -> [1, 1]
        return tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 1:
        # 1D -> [1, N]
        return tensor.unsqueeze(0)
    else:
        # Already 2D+
        return tensor
```

## Best Practices

### 1. **Document Expected Shapes**
Always document tensor shapes in comments:
```python
def forward(self, x):
    # x: [batch_size, input_features]
    x = self.layer1(x)  # -> [batch_size, hidden_size]
    x = self.layer2(x)  # -> [batch_size, output_size]
    return x
```

### 2. **Validate Early**
Check dimensions at node input, not deep in computation:
```python
async def compute(self, input):
    # Validate immediately
    self.validate_input_shape(input)
    
    # Then process
    result = self.process(input)
    return result
```

### 3. **Use Type Hints**
Include shape information in type hints:
```python
from typing import Tuple
import torch

def process_batch(
    input: torch.Tensor,  # Shape: [batch, features]
) -> Tuple[torch.Tensor, torch.Tensor]:  # Returns: ([batch, output], [batch, hidden])
    ...
```

### 4. **Test with Different Batch Sizes**
Always test nodes with batch sizes of 1, typical size, and large size:
```python
# Test cases
test_single = torch.randn(1, 128)    # Single sample
test_normal = torch.randn(32, 128)   # Normal batch
test_large = torch.randn(1024, 128)  # Large batch
```

## Special Cases

### Scalar Losses
Losses can be scalar but should document this clearly:
```python
def compute_loss(predictions, targets):
    # Returns: scalar loss (0D tensor)
    loss = F.cross_entropy(predictions, targets)
    return loss  # Scalar tensor
```

### Environment Rewards
RL rewards are the main exception to the 2D rule:
```python
# Rewards can be 1D: [num_envs]
rewards = torch.tensor([0.1, -0.2, 0.5, 1.0])  # OK for rewards

# But values should stay 2D for consistency
values = rewards.unsqueeze(1)  # [num_envs, 1]
```

### Image Data
Images have additional dimensions but still follow the convention:
```python
# Standard image batch
images = torch.randn(32, 3, 224, 224)
# [batch, channels, height, width]
#  ^^^^^ Dim 0: batch (ALWAYS)
#        ^^^^^^^^ Dim 1: features/channels
#                 ^^^^^^^^^^^ Dims 2+: spatial
```

## Migration Guide

If you have code that doesn't follow these standards:

1. **Add validation to find violations**:
```python
assert tensor.dim() >= 2, f"Found 1D tensor: {tensor.shape}"
```

2. **Fix at the source** (in templates!):
```python
# In template file
if tensor.dim() == 1:
    tensor = tensor.unsqueeze(0)
```

3. **Update tests** to expect correct dimensions

4. **Document the convention** in your node's docstring

Remember: These conventions ensure consistency, prevent bugs, and make tensor operations predictable across the entire DNNE system.