# Template variables - replaced during export
template_vars = {
    "NODE_ID": "node_6",
    "LEARNING_RATE": 0.01,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0
}

# Extract variables
NODE_ID = template_vars["NODE_ID"]
LEARNING_RATE = template_vars["LEARNING_RATE"]
MOMENTUM = template_vars["MOMENTUM"]
WEIGHT_DECAY = template_vars["WEIGHT_DECAY"]

# SGD Optimizer configuration
globals()[f"{NODE_ID}_config"] = {
    "type": "SGD",
    "lr": LEARNING_RATE,
    "momentum": MOMENTUM,
    "weight_decay": WEIGHT_DECAY
}

print(f"Configured SGD optimizer '{NODE_ID}': lr={LEARNING_RATE}, momentum={MOMENTUM}")