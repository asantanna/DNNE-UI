# Template variables - replaced during export
template_vars = {
    "NODE_ID": "node_5",
    "PREDICTIONS_VAR": "predictions",
    "LABELS_VAR": "labels"
}

# Extract variables
NODE_ID = template_vars["NODE_ID"]
PREDICTIONS_VAR = template_vars["PREDICTIONS_VAR"]
LABELS_VAR = template_vars["LABELS_VAR"]

# Cross Entropy Loss function
def compute_cross_entropy_loss(predictions, labels):
    """Compute cross entropy loss"""
    return F.cross_entropy(predictions, labels)

# Store function reference
globals()[NODE_ID] = compute_cross_entropy_loss
globals()[f"{NODE_ID}_inputs"] = {
    "predictions": PREDICTIONS_VAR,
    "labels": LABELS_VAR
}

print(f"Created cross entropy loss function '{NODE_ID}'")