# Template variables - replaced during export
template_vars = {
    "NODE_ID": "node_3",
    "INPUT_SIZE": -1,  # -1 means infer from input
    "OUTPUT_SIZE": 128,
    "ACTIVATION": "relu",
    "DROPOUT": 0.5,
    "BIAS": True,
    "WEIGHT_INIT": "xavier"
}

# Extract variables
NODE_ID = template_vars["NODE_ID"]
INPUT_SIZE = template_vars["INPUT_SIZE"]
OUTPUT_SIZE = template_vars["OUTPUT_SIZE"]
ACTIVATION = template_vars["ACTIVATION"]
DROPOUT = template_vars["DROPOUT"]
BIAS = template_vars["BIAS"]
WEIGHT_INIT = template_vars["WEIGHT_INIT"]

# Store layer configuration
globals()[f"{NODE_ID}_config"] = {
    "type": "Linear",
    "input_size": INPUT_SIZE,
    "output_size": OUTPUT_SIZE,
    "activation": ACTIVATION,
    "dropout": DROPOUT,
    "bias": BIAS,
    "weight_init": WEIGHT_INIT
}

print(f"Configured linear layer '{NODE_ID}': input_size={INPUT_SIZE} -> {OUTPUT_SIZE}, activation={ACTIVATION}")