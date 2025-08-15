# Template variables - replaced during export
template_vars = {
    "NODE_ID": None,  # e.g., "split_1"
    "CLASS_NAME": None,  # e.g., "SplitNode"
    "DIMENSION": None,  # e.g., 0
    "SPLIT_MODE": None,  # e.g., "by index" or "by size"
    "SPLIT_VALUES": None  # e.g., [10, 20, 30] for indices or [10, 10, 10, 10] for sizes
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Split node - splits a tensor into multiple outputs based on indices or sizes"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(["input"])
        self.setup_outputs(["output_a", "output_b", "output_c", "output_d"])
        
        # Configuration from template
        self.dimension = {DIMENSION}
        self.split_mode = "{SPLIT_MODE}"
        self.split_values = {SPLIT_VALUES}
        
    async def compute(self, **inputs) -> Dict[str, Any]:
        """Split the input tensor based on configured mode"""
        import torch
        import time
        
        input_tensor = inputs["input"]
        
        if input_tensor is None:
            self.node_logger.warning("Received None input, skipping")
            return {}
        
        start_time = time.time()
        
        # Get the size of the dimension we're splitting along
        if self.dimension >= input_tensor.ndim:
            raise RuntimeError(
                f"SplitNode {self.node_id}: Dimension {self.dimension} out of range for tensor with {input_tensor.ndim} dimensions"
            )
        
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
                    
                    # Create slice indices for the specific dimension
                    slice_indices = [slice(None)] * input_tensor.ndim
                    slice_indices[self.dimension] = slice(start_idx, end_idx)
                    
                    output_slice = input_tensor[tuple(slice_indices)]
                    outputs[output_name] = output_slice
                    
                    self.node_logger.debug(
                        f"Split by index: {output_name} = [{start_idx}:{end_idx}] -> shape {output_slice.shape}"
                    )
        
        elif self.split_mode == "by size":
            # Split into chunks of specific sizes
            # Example: split_values=[10, 10, 10, 10] creates 4 chunks of size 10
            
            # Filter out zero sizes and limit to 4 outputs
            non_zero_sizes = [s for s in self.split_values if s > 0][:4]
            
            if not non_zero_sizes:
                raise RuntimeError(
                    f"SplitNode {self.node_id}: No non-zero sizes provided in split_values: {self.split_values}"
                )
            
            # Verify total size doesn't exceed dimension size
            total_size = sum(non_zero_sizes)
            if total_size > dim_size:
                raise RuntimeError(
                    f"SplitNode {self.node_id}: Total split sizes {total_size} exceeds dimension size {dim_size}"
                )
            
            # Use torch.split to create the splits
            splits = torch.split(input_tensor, non_zero_sizes, dim=self.dimension)
            
            # Assign to outputs
            output_names = ["output_a", "output_b", "output_c", "output_d"]
            for i, split_tensor in enumerate(splits):
                if i < len(output_names):
                    outputs[output_names[i]] = split_tensor
                    self.node_logger.debug(
                        f"Split by size: {output_names[i]} = size {non_zero_sizes[i]} -> shape {split_tensor.shape}"
                    )
        
        else:
            raise RuntimeError(
                f"SplitNode {self.node_id}: Unknown split_mode '{self.split_mode}'"
            )
        
        self.last_compute_time = time.time() - start_time
        self.compute_count += 1
        
        self.node_logger.info(
            f"Split tensor shape {input_tensor.shape} into {len(outputs)} outputs along dimension {self.dimension}"
        )
        
        return outputs