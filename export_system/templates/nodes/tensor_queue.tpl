# Template variables - replaced during export
template_vars = {
    "NODE_ID": "tensor_1",
    "CLASS_NAME": "TensorNode",
    "TENSOR_DIMS": "[10]",
    "FILL_MODE": "zeros",
    "CUSTOM_FILL": "0.0",
    "DTYPE": "float32",
    "SEED": "-1"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Tensor Constant Node - generates constant tensors with configurable initialization"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        # No inputs for constant generator
        self.setup_inputs(required=[])
        self.setup_outputs(["tensor"])
        
        # Parse configuration
        self.dims = self._parse_dims("{TENSOR_DIMS}")
        self.fill_mode = "{FILL_MODE}"
        self.custom_fill = {CUSTOM_FILL}
        self.dtype = self._get_torch_dtype("{DTYPE}")
        self.seed = {SEED}
        
        # Set random seed if specified
        if self.seed >= 0:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
        
        # Pre-generate tensor if not using random initialization
        self.cached_tensor = None
        if self.fill_mode in ["zeros", "ones", "custom"]:
            self.cached_tensor = self._generate_tensor()
            
        # Set queue size to 2 for immediate availability
        self.output_queues["tensor"] = asyncio.Queue(maxsize=2)
    
    def _parse_dims(self, dims_str: str) -> tuple:
        """Parse dimension string into tuple of integers"""
        import re
        
        # Remove brackets if present
        dims_str = dims_str.strip().strip("[]")
        
        # Split by comma and convert to integers
        if ',' in dims_str:
            dims = [int(d.strip()) for d in dims_str.split(',')]
        else:
            dims = [int(dims_str.strip())]
        
        return tuple(dims)
    
    def _get_torch_dtype(self, dtype_str: str):
        """Convert string dtype to torch dtype"""
        import torch
        
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool
        }
        return dtype_map.get(dtype_str, torch.float32)
    
    def _generate_tensor(self) -> torch.Tensor:
        """Generate tensor based on configuration"""
        import torch
        import torch.nn.init as init
        
        # Set seed for this generation if specified
        if self.seed >= 0:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
        
        # Create base tensor
        if self.fill_mode == "zeros":
            tensor = torch.zeros(self.dims, dtype=self.dtype)
        elif self.fill_mode == "ones":
            tensor = torch.ones(self.dims, dtype=self.dtype)
        elif self.fill_mode == "custom":
            tensor = torch.full(self.dims, self.custom_fill, dtype=self.dtype)
        elif self.fill_mode == "uniform":
            # Uniform distribution [-1, 1]
            tensor = torch.empty(self.dims, dtype=self.dtype)
            init.uniform_(tensor, a=-1.0, b=1.0)
        elif self.fill_mode == "normal":
            # Standard normal distribution
            tensor = torch.empty(self.dims, dtype=self.dtype)
            init.normal_(tensor, mean=0.0, std=1.0)
        elif self.fill_mode == "kaiming_normal":
            # Kaiming/He normal initialization
            tensor = torch.empty(self.dims, dtype=self.dtype)
            # For tensors without explicit fan_in/fan_out, use first dimension
            if len(self.dims) >= 2:
                init.kaiming_normal_(tensor, mode='fan_out', nonlinearity='relu')
            else:
                # For 1D tensors, use normal distribution with adjusted std
                init.normal_(tensor, mean=0.0, std=2.0 / (self.dims[0] ** 0.5))
        elif self.fill_mode == "kaiming_uniform":
            # Kaiming/He uniform initialization
            tensor = torch.empty(self.dims, dtype=self.dtype)
            if len(self.dims) >= 2:
                init.kaiming_uniform_(tensor, mode='fan_out', nonlinearity='relu')
            else:
                # For 1D tensors, use uniform with adjusted range
                bound = (6.0 / self.dims[0]) ** 0.5
                init.uniform_(tensor, a=-bound, b=bound)
        elif self.fill_mode == "xavier_normal":
            # Xavier/Glorot normal initialization
            tensor = torch.empty(self.dims, dtype=self.dtype)
            if len(self.dims) >= 2:
                init.xavier_normal_(tensor)
            else:
                # For 1D tensors, use normal with adjusted std
                init.normal_(tensor, mean=0.0, std=(2.0 / self.dims[0]) ** 0.5)
        elif self.fill_mode == "xavier_uniform":
            # Xavier/Glorot uniform initialization
            tensor = torch.empty(self.dims, dtype=self.dtype)
            if len(self.dims) >= 2:
                init.xavier_uniform_(tensor)
            else:
                # For 1D tensors, use uniform with adjusted range
                bound = (3.0 / self.dims[0]) ** 0.5
                init.uniform_(tensor, a=-bound, b=bound)
        else:
            # Default to zeros if unknown mode
            self.node_logger.warning(f"Unknown fill_mode '{self.fill_mode}', defaulting to zeros")
            tensor = torch.zeros(self.dims, dtype=self.dtype)
        
        return tensor
    
    async def run(self):
        """Custom run method for constant generation"""
        import asyncio
        import time
        
        self.running = True
        self.node_logger.info(f"Starting Tensor node {self.node_id} - dims: {self.dims}, mode: {self.fill_mode}")
        
        try:
            generation_count = 0
            while self.running:
                # Generate or reuse tensor
                if self.cached_tensor is not None:
                    # Reuse cached tensor for deterministic modes
                    tensor = self.cached_tensor.clone()
                else:
                    # Generate new tensor for random modes
                    tensor = self._generate_tensor()
                
                generation_count += 1
                
                # Log generation info periodically
                if generation_count % 10 == 1:
                    self.node_logger.debug(
                        f"Generated tensor #{generation_count} - "
                        f"shape: {tensor.shape}, dtype: {tensor.dtype}, "
                        f"min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}"
                    )
                
                # Send to output queue (will block if queue is full)
                await self.send_output("tensor", tensor)
                
        except asyncio.CancelledError:
            self.node_logger.info(f"Tensor Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False
    
    async def compute(self, **inputs) -> Dict[str, Any]:
        """Required by abstract base class - not used since we override run()"""
        # This method is required by QueueNode abstract base class
        # but not actually called since we override run() method
        raise NotImplementedError("Tensor node uses custom run() method, not compute()")