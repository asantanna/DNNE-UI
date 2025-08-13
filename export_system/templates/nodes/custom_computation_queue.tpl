# Template variables - replaced during export
template_vars = {
    "NODE_ID": "custom_1",
    "CLASS_NAME": "CustomComputationNode",
    "SRC_PATH": "/path/to/custom.py",
    "MODULE_NAME": "custom_module"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Custom computation node that loads and executes user-defined functions"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
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
        if len(params) != 1:
            raise ValueError(f"compute() function must accept exactly 1 parameter (input tensor), got {len(params)}")
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    async def compute(self, input) -> Dict[str, Any]:
        """Execute the custom computation function"""
        try:
            # Ensure input is on correct device
            input_tensor = input.to(self.device)
            
            # Call the custom compute function
            output = self.compute_fn(input_tensor)
            
            # Ensure output is a tensor
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"compute() must return a torch.Tensor, got {type(output)}")
            
            return {"output": output}
            
        except Exception as e:
            self.logger.error(f"Error in custom computation: {e}")
            raise RuntimeError(f"Custom computation failed: {e}") from e