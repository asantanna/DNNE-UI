# Template variables - replaced during export
template_vars = {
    "NODE_ID": "node_8",
    "LOSS_VAR": "loss",
    "OPTIMIZER_VAR": "optimizer"
}

# Extract variables
NODE_ID = template_vars["NODE_ID"]
LOSS_VAR = template_vars["LOSS_VAR"]
OPTIMIZER_VAR = template_vars["OPTIMIZER_VAR"]

# Training step configuration
globals()[f"{NODE_ID}_config"] = {
    "type": "TrainingStep",
    "loss_var": LOSS_VAR,
    "optimizer_var": OPTIMIZER_VAR
}

print(f"Configured training step '{NODE_ID}'")