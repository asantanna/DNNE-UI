# Template variables - replaced during export
template_vars = {
    "NODE_ID": None,  # e.g., "split_1"
    "CLASS_NAME": None,  # e.g., "SplitNode"
    "DIMENSION": 1,  # ALWAYS 1 per tensor standards (feature dimension)
    "SPLIT_MODE": None,  # e.g., "by index", "by size", or "by name"
    "SPLIT_VALUES": None  # e.g., [10, 20, 30] for indices or [10, 10, 10, 10] for sizes
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Split node - splits a tensor along feature dimension into multiple outputs"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(["input"])
        self.setup_outputs(["output_a", "output_b", "output_c", "output_d"])
        
        # Configuration from template
        self.dimension = {DIMENSION}  # Per tensor standards: always dim=1 for features
        self.split_mode = "{SPLIT_MODE}"
        self.split_values = {SPLIT_VALUES}
        
    async def compute(self, **inputs) -> Dict[str, Any]:
        """Split the input tensor along feature dimension based on configured mode"""
        import torch
        import time
        
        input_tensor = inputs["input"]
        
        if input_tensor is None:
            self.node_logger.warning("Received None input, skipping")
            return {}
        
        start_time = time.time()
        
        # Get the size of the feature dimension (dim=1)
        dim_size = input_tensor.shape[self.dimension]
        
        # Prepare outputs dict
        outputs = {}
        
        if self.split_mode == "by index":
            # Split at specific indices
            # Example: split_values=[10, 20, 30] creates splits [0:10], [10:20], [20:30], [30:]
            indices = [0] + self.split_values + [dim_size]
            
            # Create slices
            for i, output_name in enumerate(["output_a", "output_b", "output_c", "output_d"]):
                if i < len(indices) - 1:
                    start_idx = indices[i]
                    end_idx = indices[i + 1]
                    
                    if start_idx >= dim_size:
                        # This output and all subsequent ones will be empty
                        break
                    
                    # Clamp end_idx to dim_size
                    end_idx = min(end_idx, dim_size)
                    
                    # Create slice indices for the feature dimension
                    slice_indices = [slice(None)] * input_tensor.ndim
                    slice_indices[self.dimension] = slice(start_idx, end_idx)
                    
                    output_slice = input_tensor[tuple(slice_indices)]
                    outputs[output_name] = output_slice
        
        elif self.split_mode == "by size":
            # Split into chunks of specific sizes
            # Filter out zero sizes and limit to 4 outputs
            non_zero_sizes = [s for s in self.split_values if s > 0][:4]
            
            # Use torch.split to create the splits along feature dimension
            splits = torch.split(input_tensor, non_zero_sizes, dim=self.dimension)
            
            # Assign to outputs
            output_names = ["output_a", "output_b", "output_c", "output_d"]
            for i, split_tensor in enumerate(splits):
                if i < len(output_names):
                    outputs[output_names[i]] = split_tensor
        
        elif self.split_mode == "by name":
            # When semantic names are used, split_values contains resolved ranges
            # Example: split_values=[[1, 2], [5, 7]] extracts features [1:2] and [5:7]
            
            output_names = ["output_a", "output_b", "output_c", "output_d"]
            
            # Check if split_values contains ranges (lists)
            if self.split_values and isinstance(self.split_values[0], list):
                # Resolved ranges from semantic names
                for i, range_pair in enumerate(self.split_values):
                    if i >= len(output_names):
                        break
                    
                    start_idx, end_idx = range_pair
                    
                    # Skip if range starts beyond dimension size
                    if start_idx >= dim_size:
                        continue
                    
                    # Clamp end_idx to dim_size
                    end_idx = min(end_idx, dim_size)
                    
                    # Create slice indices for the feature dimension
                    slice_indices = [slice(None)] * input_tensor.ndim
                    slice_indices[self.dimension] = slice(start_idx, end_idx)
                    
                    output_slice = input_tensor[tuple(slice_indices)]
                    outputs[output_names[i]] = output_slice
        
        self.last_compute_time = time.time() - start_time
        self.compute_count += 1
        
        return outputs