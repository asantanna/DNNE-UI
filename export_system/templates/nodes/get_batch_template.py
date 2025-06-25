# Template variables - replaced during export
template_vars = {
    "NODE_ID": "node_3",
    "DATALOADER_VAR": "node_2_output",
    "CONTEXT_VAR": "context"
}

# Extract variables
NODE_ID = template_vars["NODE_ID"]
DATALOADER_VAR = template_vars["DATALOADER_VAR"]
CONTEXT_VAR = template_vars["CONTEXT_VAR"]

# Get Batch configuration
# This node represents getting batches in the training loop
globals()[f"{NODE_ID}_config"] = {
    "dataloader_var": DATALOADER_VAR,
    "outputs": ["batch_images", "batch_labels"]
}

print(f"Configured batch getter '{NODE_ID}' from dataloader '{DATALOADER_VAR}'")