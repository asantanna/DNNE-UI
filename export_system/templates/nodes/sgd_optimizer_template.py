# Template variables - replaced during export
template_vars = {
    "NODE_ID": "optimizer_1",
    "LEARNING_RATE": 0.01,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0
}

# SGD Optimizer: {NODE_ID}
# Note: Optimizer creation is handled in the training loop
# This template stores the configuration
optimizer_config_{NODE_ID} = {{
    "lr": template_vars["LEARNING_RATE"],
    "momentum": template_vars["MOMENTUM"],
    "weight_decay": template_vars["WEIGHT_DECAY"]
}}
