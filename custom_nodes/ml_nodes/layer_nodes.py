"""
Neural network layer nodes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import cleandoc
from .base import RoboticsNodeBase, get_context


class NetworkNode(RoboticsNodeBase):
    """
    Network Node
    Consolidates multiple LinearLayer nodes into a single PyTorch Sequential model with checkpoint support.
    For checkpoint debugging: check console logs or exported code for actual node ID.
    """

    def __init__(self):
        super().__init__()
        self.checkpoint_manager = None
        self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR", {"tooltip": "Input tensor to process through the neural network"}),
                "to_output": ("TENSOR", {"tooltip": "Loop-back connection from the last layer output"}),
                # Checkpoint parameters - must be widgets to save to widgets_values
                "checkpoint_enabled": ("BOOLEAN", {"default": True, "widget": {"name": "checkpoint_enabled"}, "tooltip": "Enable automatic checkpoint saving for this network. Checkpoints saved to 'node_<ID>' subdirectories."}),
                "checkpoint_trigger_type": (["epoch", "time", "best_metric"], {"default": "epoch", "widget": {"name": "checkpoint_trigger_type"}, "tooltip": "When to save checkpoints: every N steps, time intervals, or metric improvements"}),
                "checkpoint_trigger_value": ("STRING", {"default": "50", "widget": {"name": "checkpoint_trigger_value"}, "tooltip": "Trigger value: number (steps), time format (1h30m), or 'min'/'max' (metrics)"}),
                "checkpoint_load_on_start": ("BOOLEAN", {"default": False, "widget": {"name": "checkpoint_load_on_start"}, "tooltip": "Automatically load saved checkpoint when network starts"}),
            },
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "MODEL")
    RETURN_NAMES = ("layers", "output", "model")
    FUNCTION = "forward"
    CATEGORY = "ml"
    DESCRIPTION = cleandoc(__doc__)

    def forward(self, input, to_output, checkpoint_enabled=False,
                checkpoint_trigger_type="epoch", checkpoint_trigger_value="50", 
                checkpoint_load_on_start=False, unique_id=None):
        # Log actual node ID for checkpoint debugging  
        actual_node_id = unique_id or f"network_{id(self)}"
        if checkpoint_enabled:
            print(f"🔍 Network Node ID: {actual_node_id} (look for 'node_{actual_node_id}' in checkpoint directories)")
        
        # Initialize checkpoint manager if needed
        if checkpoint_enabled and self.checkpoint_manager is None:
            # Import here to avoid circular imports in export
            from ..export_system.templates.base.run_utils import CheckpointManager, validate_checkpoint_config
            
            # Validate checkpoint configuration
            checkpoint_config = {
                'enabled': checkpoint_enabled,
                'trigger_type': checkpoint_trigger_type,
                'trigger_value': checkpoint_trigger_value
            }
            try:
                validate_checkpoint_config(checkpoint_config)
            except ValueError as e:
                print(f"⚠️ Checkpoint configuration error: {e}")
                checkpoint_enabled = False
            
            if checkpoint_enabled:
                # Get checkpoint directory from command line args (set by runner.py)
                try:
                    import builtins
                    save_checkpoint_dir = getattr(builtins, 'SAVE_CHECKPOINT_DIR', None)
                    load_checkpoint_dir = getattr(builtins, 'LOAD_CHECKPOINT_DIR', None)
                except:
                    save_checkpoint_dir = None
                    load_checkpoint_dir = None
                    
                self.checkpoint_manager = CheckpointManager(
                    node_id=actual_node_id,
                    checkpoint_dir=save_checkpoint_dir
                )
                
                # Load checkpoint on start if requested
                if checkpoint_load_on_start and load_checkpoint_dir:
                    if hasattr(self, 'model') and self.model is not None:
                        self.load_checkpoint(self.model, load_checkpoint_dir)
        
        # This is just for UI - actual implementation happens in export
        # The network structure is defined by the connected layers
        # Return: (layers, output, model)
        # The model output can be connected to SGD optimizer
        return (None, to_output, self)  # layers output is just for connectivity, self represents the model
    
    def save_checkpoint(self, model, trigger_type="external", trigger_value=None, 
                       current_epoch=None, current_metric=None, metadata=None):
        """
        Save model checkpoint
        
        Args:
            model: PyTorch model to save
            trigger_type: Type of trigger ('epoch', 'time', 'best_metric', 'external')
            trigger_value: Value for the trigger (depends on type)
            current_epoch: Current epoch number (for epoch-based triggers)
            current_metric: Current metric value (for best metric triggers)
            metadata: Additional metadata to include
            
        Returns:
            str: Path to saved checkpoint file, or None if not saved
        """
        if not self.checkpoint_manager:
            print("⚠️ No checkpoint manager initialized")
            return None
        
        # Check if we should checkpoint
        should_checkpoint = self.checkpoint_manager.should_checkpoint(
            trigger_type, trigger_value, current_epoch, current_metric
        )
        
        if should_checkpoint:
            # Prepare metadata with model information
            checkpoint_metadata = {
                'trigger_type': trigger_type,
                'trigger_value': trigger_value,
                'current_epoch': current_epoch,
                'current_metric': current_metric,
                'model_type': type(model).__name__,
                'architecture': getattr(model, 'architecture_info', None),
                'model_info': {
                    'num_parameters': sum(p.numel() for p in model.parameters()),
                    'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
                }
            }
            
            if metadata:
                checkpoint_metadata.update(metadata)
            
            # Save checkpoint (only model weights + metadata)
            success = self.checkpoint_manager.save_checkpoint(
                model.state_dict(), metadata=checkpoint_metadata
            )
            return success
        
        return None
    
    def load_checkpoint(self, model, load_checkpoint_dir=None):
        """
        Load model checkpoint
        
        Args:
            model: PyTorch model to load state into
            load_checkpoint_dir: Override load directory (from command line)
            
        Returns:
            bool: True if checkpoint loaded successfully
        """
        if not self.checkpoint_manager:
            print("⚠️ No checkpoint manager initialized")
            return False
        
        # Load checkpoint from command line directory or default
        checkpoint_data = self.checkpoint_manager.load_checkpoint(load_checkpoint_dir)
        if not checkpoint_data:
            print("⚠️ No checkpoint found to load")
            return False
        
        try:
            # Load model state
            model.load_state_dict(checkpoint_data['model_state_dict'])
            
            # Print loaded info
            metadata = checkpoint_data.get('metadata', {})
            epoch = metadata.get('current_epoch', 'unknown')
            metric = metadata.get('current_metric', 'unknown')
            
            print(f"✅ Model checkpoint loaded - epoch: {epoch}, metric: {metric}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error loading model checkpoint: {e}")
            return False


class LinearLayerNode(RoboticsNodeBase):
    """
    Linear Layer Node
    Fully connected layer with configurable activation and dropout.
    """
    
    DESCRIPTION = cleandoc(__doc__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR", {"tooltip": "Input tensor to transform (automatically flattened if > 2D)"}),
                "output_size": ("INT", {"default": 128, "min": 1, "max": 4096, "tooltip": "Number of output features (neurons) in this layer"}),
                "bias": ("BOOLEAN", {"default": True, "tooltip": "Whether to include learnable bias parameters"}),
                "activation": (["none", "relu", "tanh", "sigmoid"], {"default": "relu", "tooltip": "Activation function to apply after linear transformation"}),
                "dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "tooltip": "Dropout probability for regularization (0.0 = no dropout)"}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "forward"
    CATEGORY = "ml"

    def forward(self, input, output_size, bias, activation, dropout):
        context = get_context()
        
        # Flatten input if needed
        if len(input.shape) > 2:
            input = input.view(input.size(0), -1)

        input_size = input.shape[1]
        layer_key = f"linear_{id(self)}_{input_size}_{output_size}"

        # Get or create layer
        if layer_key not in context.memory:
            layer = nn.Linear(input_size, output_size, bias=bias)
            if torch.cuda.is_available() and input.is_cuda:
                layer = layer.cuda()
            context.memory[layer_key] = layer
        else:
            layer = context.memory[layer_key]

        # Set training mode
        layer.train(context.training)

        # Forward pass
        output = layer(input)

        # Apply activation
        if activation == "relu":
            output = F.relu(output)
        elif activation == "tanh":
            output = torch.tanh(output)
        elif activation == "sigmoid":
            output = torch.sigmoid(output)

        # Apply dropout
        if dropout > 0 and context.training:
            output = F.dropout(output, p=dropout, training=True)

        return (output,)


class Conv2DLayerNode(RoboticsNodeBase):
    """
    Conv2D Layer Node
    2D convolutional layer with configurable kernel, stride, and padding.
    """
    
    DESCRIPTION = cleandoc(__doc__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR", {"tooltip": "Input tensor in format (batch, channels, height, width)"}),
                "out_channels": ("INT", {"default": 32, "min": 1, "max": 512, "tooltip": "Number of output channels (feature maps) to produce"}),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 11, "tooltip": "Size of convolution kernel (e.g., 3 for 3x3, 5 for 5x5)"}),
                "stride": ("INT", {"default": 1, "min": 1, "max": 5, "tooltip": "Step size for moving the kernel (1 = no downsampling)"}),
                "padding": ("INT", {"default": 1, "min": 0, "max": 5, "tooltip": "Zero-padding around input borders (1 preserves size with kernel=3)"}),
                "activation": (["none", "relu", "tanh", "sigmoid"], {"default": "relu", "tooltip": "Activation function to apply after convolution"}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "forward"
    CATEGORY = "ml"

    def forward(self, input, out_channels, kernel_size, stride, padding, activation):
        context = get_context()
        
        # Get input channels
        in_channels = input.shape[1]
        layer_key = f"conv2d_{id(self)}_{in_channels}_{out_channels}_{kernel_size}"

        # Get or create layer
        if layer_key not in context.memory:
            layer = nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            if torch.cuda.is_available() and input.is_cuda:
                layer = layer.cuda()
            context.memory[layer_key] = layer
        else:
            layer = context.memory[layer_key]

        # Set training mode
        layer.train(context.training)

        # Forward pass
        output = layer(input)

        # Apply activation
        if activation == "relu":
            output = F.relu(output)
        elif activation == "tanh":
            output = torch.tanh(output)
        elif activation == "sigmoid":
            output = torch.sigmoid(output)

        return (output,)


class ActivationNode(RoboticsNodeBase):
    """
    Activation Node
    Applies activation functions like ReLU, Sigmoid, Tanh, or Softmax.
    """
    
    DESCRIPTION = cleandoc(__doc__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR", {"tooltip": "Input tensor to apply activation function to"}),
                "activation": (["relu", "tanh", "sigmoid", "softmax", "leaky_relu", "elu"], {"default": "relu", "tooltip": "Activation function: relu (most common), sigmoid (0-1), tanh (-1 to 1), softmax (probabilities)"}),
                "negative_slope": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "tooltip": "Negative slope for leaky_relu activation (0.01 is standard)"}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "apply"
    CATEGORY = "ml"

    def apply(self, input, activation, negative_slope):
        if activation == "relu":
            output = F.relu(input)
        elif activation == "tanh":
            output = torch.tanh(input)
        elif activation == "sigmoid":
            output = torch.sigmoid(input)
        elif activation == "softmax":
            output = F.softmax(input, dim=-1)
        elif activation == "leaky_relu":
            output = F.leaky_relu(input, negative_slope=negative_slope)
        elif activation == "elu":
            output = F.elu(input)
        
        return (output,)


class DropoutNode(RoboticsNodeBase):
    """
    Dropout Node
    Applies dropout regularization during training to prevent overfitting.
    """
    
    DESCRIPTION = cleandoc(__doc__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR", {"tooltip": "Input tensor to apply dropout regularization to"}),
                "dropout_rate": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.9, "tooltip": "Probability of setting elements to zero (0.5 = 50% dropout)"}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "apply"
    CATEGORY = "ml"

    def apply(self, input, dropout_rate):
        context = get_context()
        if context.training and dropout_rate > 0:
            output = F.dropout(input, p=dropout_rate, training=True)
        else:
            output = input
        return (output,)


class BatchNormNode(RoboticsNodeBase):
    """
    Batch Normalization Node
    Normalizes inputs to improve training stability and convergence.
    """
    
    DESCRIPTION = cleandoc(__doc__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR", {"tooltip": "Input tensor (2D for BatchNorm1d, 4D for BatchNorm2d)"}),
                "momentum": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.99, "tooltip": "Momentum for running mean/variance updates (0.1 is standard)"}),
                "eps": ("FLOAT", {"default": 1e-5, "min": 1e-8, "max": 1e-3, "tooltip": "Small value for numerical stability (prevents division by zero)"}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "apply"
    CATEGORY = "ml"

    def apply(self, input, momentum, eps):
        context = get_context()
        
        # Determine the number of features
        if len(input.shape) == 2:  # Fully connected layer output
            num_features = input.shape[1]
            layer_key = f"batchnorm1d_{id(self)}_{num_features}"
            layer_class = nn.BatchNorm1d
        elif len(input.shape) == 4:  # Convolutional layer output
            num_features = input.shape[1]
            layer_key = f"batchnorm2d_{id(self)}_{num_features}"
            layer_class = nn.BatchNorm2d
        else:
            raise ValueError(f"Unsupported input shape: {input.shape}")

        # Get or create layer
        if layer_key not in context.memory:
            layer = layer_class(num_features, momentum=momentum, eps=eps)
            if torch.cuda.is_available() and input.is_cuda:
                layer = layer.cuda()
            context.memory[layer_key] = layer
        else:
            layer = context.memory[layer_key]

        # Set training mode
        layer.train(context.training)

        # Apply batch norm
        output = layer(input)
        return (output,)


class FlattenNode(RoboticsNodeBase):
    """
    Flatten Node
    Flattens multi-dimensional tensors for fully connected layers.
    """
    
    DESCRIPTION = cleandoc(__doc__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR", {"tooltip": "Input tensor to flatten into fewer dimensions"}),
                "start_dim": ("INT", {"default": 1, "min": 0, "max": 3, "tooltip": "First dimension to flatten (0=batch, 1=preserve batch)"}),
                "end_dim": ("INT", {"default": -1, "min": -1, "max": 3, "tooltip": "Last dimension to flatten (-1=all remaining dimensions)"}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "flatten"
    CATEGORY = "ml"

    def flatten(self, input, start_dim, end_dim):
        output = torch.flatten(input, start_dim, end_dim)
        return (output,)