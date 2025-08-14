# Custom Compute Functions

This directory is the standard location for Python scripts used by the Custom Computation node in DNNE.

## Usage

Place your custom Python files here and reference them in the Custom Computation node by filename only (no path required).

## Required Function Signature

Every Python file must contain a `compute` function with the following signature:

```python
import torch
from typing import Optional

def compute(input: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Process the input tensor and return a result tensor or None.
    
    Args:
        input: Input tensor from the previous node
        
    Returns:
        - torch.Tensor: Processed output tensor
        - None: No output (acts as a filter)
    """
    # Your custom computation here
    return output  # or return None to filter
```

## Examples

### Simple Transform
```python
def compute(input):
    # Scale input by 2 and add 1
    return input * 2 + 1
```

### Shape Preserving Operation
```python
def compute(input):
    # Apply sigmoid activation
    return torch.sigmoid(input)
```

### Reduction Operation
```python
def compute(input):
    # Return mean across batch dimension
    return input.mean(dim=0, keepdim=True)
```

### Filter Operation
```python
def compute(input):
    # Only pass through tensors with positive mean
    if input.mean() > 0:
        return input
    else:
        return None  # Filter out this input
```

## File Naming

- Use descriptive names: `normalize_features.py`, `apply_custom_activation.py`
- Avoid spaces in filenames (use underscores instead)
- Must be valid Python module names

## Path Resolution

When you specify just a filename in the Custom Computation node (e.g., `"my_function.py"`), DNNE will automatically look in this directory.

You can also specify:
- Relative paths: `"./local/my_function.py"`
- Absolute paths: `"/home/user/scripts/my_function.py"`