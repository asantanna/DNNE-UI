# Template variables - replaced during export
template_vars = {
    "NODE_ID": "node_3",
    "INPUT_VAR": "input_tensor",
    "OUTPUT_SIZE": 128,
    "ACTIVATION": "relu",
    "DROPOUT": 0.5,
    "BIAS": True,
    "WEIGHT_INIT": "xavier"
}

# Extract variables for cleaner code
NODE_ID = template_vars["NODE_ID"]
INPUT_VAR = template_vars["INPUT_VAR"]
OUTPUT_SIZE = template_vars["OUTPUT_SIZE"]
ACTIVATION = template_vars["ACTIVATION"]
DROPOUT = template_vars["DROPOUT"]
BIAS = template_vars["BIAS"]
WEIGHT_INIT = template_vars["WEIGHT_INIT"]

# Linear Layer
if NODE_ID not in context.memory:
    # Get input tensor
    input_tensor = globals()[INPUT_VAR]
    
    # Determine input size
    if input_tensor.dim() > 2:
        input_size = input_tensor.view(input_tensor.size(0), -1).shape[1]
    else:
        input_size = input_tensor.shape[1]
    
    # Create layer
    layer = nn.Linear(input_size, OUTPUT_SIZE, bias=BIAS)
    
    # Initialize weights
    if WEIGHT_INIT == "xavier":
        nn.init.xavier_uniform_(layer.weight)
    elif WEIGHT_INIT == "kaiming":
        nn.init.kaiming_uniform_(layer.weight)
    elif WEIGHT_INIT == "normal":
        nn.init.normal_(layer.weight, std=0.02)
    
    if BIAS:
        nn.init.zeros_(layer.bias)
    
    context.memory[NODE_ID] = layer
else:
    layer = context.memory[NODE_ID]

# Get input and compute forward pass
input_tensor = globals()[INPUT_VAR]
if input_tensor.dim() > 2:
    input_tensor = input_tensor.view(input_tensor.size(0), -1)

output = layer(input_tensor)

# Apply activation
if ACTIVATION == "relu":
    output = F.relu(output)
elif ACTIVATION == "sigmoid":
    output = torch.sigmoid(output)
elif ACTIVATION == "tanh":
    output = torch.tanh(output)

# Apply dropout
if DROPOUT > 0 and context.training:
    output = F.dropout(output, p=DROPOUT)

# Make output available
globals()[NODE_ID] = layer  # Store the layer
globals()[f"{NODE_ID}_output"] = output  # Store the output
