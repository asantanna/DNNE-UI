# Custom Compute Functions

This directory is the standard location for Python scripts used by the Custom Computation node in DNNE.

## Usage

Place your custom Python files here and reference them in the Custom Computation node by filename only (no path required).

## Required Function Signature

Every Python file must contain a `compute` function with the following signature:

```python
import torch
from typing import Optional

def compute(input: torch.Tensor, extra_args=None) -> Optional[torch.Tensor]:
    """
    Process the input tensor and return a result tensor or None.
    
    Args:
        input: Input tensor from the previous node
        extra_args: Optional additional input (tensor or other data) from connected node
        
    Returns:
        - torch.Tensor: Processed output tensor
        - None: No output (acts as a filter)
    """
    # Your custom computation here
    return output  # or return None to filter
```

## Configuration Support (Optional)

Your script can optionally support configuration by implementing:

```python
# Global config storage
config = {}

def set_config_info(cfg):
    """Called by CustomComputation node to set configuration."""
    global config
    config = cfg
    print(f"Config set: {config}")
```

## Examples

### Simple Transform
```python
def compute(input, extra_args=None):
    # Scale input by 2 and add 1
    return input * 2 + 1
```

### Using Extra Args
```python
def compute(input, extra_args=None):
    # Combine input with extra_args if provided
    if extra_args is not None:
        return input + extra_args
    return input
```

### Shape Preserving Operation
```python
def compute(input, extra_args=None):
    # Apply sigmoid activation
    return torch.sigmoid(input)
```

### Reduction Operation
```python
def compute(input, extra_args=None):
    # Return mean across batch dimension
    return input.mean(dim=0, keepdim=True)
```

### Filter Operation
```python
def compute(input, extra_args=None):
    # Only pass through tensors with positive mean
    if input.mean() > 0:
        return input
    else:
        return None  # Filter out this input
```

### Using Configuration
```python
config = {}

def set_config_info(cfg):
    global config
    config = cfg

def compute(input, extra_args=None):
    # Use config to control behavior
    scale = config.get("scale", 1.0)
    return input * scale
```

### Augmenting with Extra Data (e.g., debug visualization)
```python
config = {}

def set_config_info(cfg):
    global config
    config = cfg

def compute(action, extra_args=None):
    """Augment action tensor with debug data."""
    if config.get("DEBUG", False) and extra_args is not None:
        # Attach extra data to action for downstream processing
        key_name = config.get("key_name", "data")
        action.extra_args = {
            key_name: extra_args.flatten().tolist()
        }
    return action
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

## Important Notes

1. **Extra Args Parameter**: The `extra_args` parameter is optional. If the CustomComputation node has an `extra_args` input connected, it will be passed to your function. Otherwise, it will be `None`.

2. **Config Dictionary**: If your script implements `set_config_info()`, the CustomComputation node will call it with the config dictionary specified in the node's config widget.

3. **Fail-Fast Principle**: When using config, it's recommended to fail fast with clear error messages rather than using defaults:
   ```python
   if "required_key" not in config:
       raise RuntimeError("config must contain 'required_key'")
   ```

4. **Attaching Extra Data**: You can attach additional data to tensors using the `extra_args` attribute. This is useful for passing metadata or debug information through the graph.