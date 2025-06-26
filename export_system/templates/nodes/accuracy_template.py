# Template variables - replaced during export
template_vars = {
    "NODE_ID": "node_7",
    "PREDICTIONS_VAR": "predictions",
    "LABELS_VAR": "labels"
}

# Extract variables
NODE_ID = template_vars["NODE_ID"]
PREDICTIONS_VAR = template_vars["PREDICTIONS_VAR"]
LABELS_VAR = template_vars["LABELS_VAR"]

# Accuracy calculation function
def calculate_accuracy(predictions, labels):
    """Calculate classification accuracy"""
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total

# Store function reference
globals()[NODE_ID] = calculate_accuracy
globals()[f"{NODE_ID}_inputs"] = {
    "predictions": PREDICTIONS_VAR,
    "labels": LABELS_VAR
}

print(f"Created accuracy function '{NODE_ID}'")