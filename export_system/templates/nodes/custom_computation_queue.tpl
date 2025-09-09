# Template variables - replaced during export
template_vars = {
    "NODE_ID": "custom_1",
    "CLASS_NAME": "CustomComputationNode",
    "SRC_PATH": "/path/to/custom.py",
    "MODULE_NAME": "custom_module",
    "CONFIG": "{}"  # Python dict as string, e.g. {"DEBUG": True, "scale": 2.0}
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Custom computation node that loads and executes user-defined functions"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        # Set up inputs based on whether extra_args is connected
        if {HAS_EXTRA_ARGS}:
            self.setup_inputs(required=["input", "extra_args"])
        else:
            self.setup_inputs(required=["input"])
        self.setup_outputs(["output"])
        
        # Import the custom module
        src_path = "{SRC_PATH}"
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Custom computation file not found: {src_path}")
        
        # Load the module dynamically
        import importlib.util
        spec = importlib.util.spec_from_file_location("{MODULE_NAME}_{NODE_ID}", src_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module from: {src_path}")
            
        self.custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.custom_module)
        
        # Verify the compute function exists and has correct signature
        if not hasattr(self.custom_module, 'compute'):
            raise AttributeError(f"Module {src_path} must contain a 'compute' function")
        
        self.compute_fn = self.custom_module.compute
        
        # Verify function signature (basic check)
        import inspect
        sig = inspect.signature(self.compute_fn)
        params = list(sig.parameters.keys())
        if len(params) < 1 or len(params) > 2:
            raise ValueError(f"compute() function must accept 1-2 parameters (input tensor, optional extra_args), got {len(params)}")
        
        # Set config if the script supports it
        config = {CONFIG}  # This gets substituted during export
        if hasattr(self.custom_module, 'set_config_info'):
            try:
                self.custom_module.set_config_info(config)
            except Exception as e:
                raise RuntimeError(f"Failed to set config for {src_path}: {e}")
        
        # Get device from global configuration
        from framework.globals import Global as g
        self.device = torch.device(g.get_device())
        
    async def compute(self, input, extra_args=None) -> Dict[str, Any]:
        """Execute the custom computation function"""
        try:
            # Ensure input is on correct device
            input_tensor = input.to(self.device)
            
            # Call the custom compute function with optional extra_args
            # Gradients are preserved automatically if input has them
            output = self.compute_fn(input_tensor, extra_args)
            
            # Handle None return (filter mode - no output emitted)
            if output is None:
                return {}  # Empty dict means no output
            
            # Let PyTorch fail-fast if output is not a tensor
            return {"output": output}
            
        except CauseExitException:
            # Let CauseExitException propagate for graceful exits
            raise
        except Exception as e:
            self.node_logger.error(f"Error in custom computation: {e}")
            raise RuntimeError(f"Custom computation failed: {e}") from e