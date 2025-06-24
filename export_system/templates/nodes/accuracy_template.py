# Template variables - replaced during export
template_vars = {
    "NODE_ID": "accuracy_1",
    "PREDICTIONS_VAR": "predictions",
    "LABELS_VAR": "labels"
}

# Calculate Accuracy: {NODE_ID}
_, predicted = torch.max(eval(template_vars["PREDICTIONS_VAR"]), 1)
correct = (predicted == eval(template_vars["LABELS_VAR"])).sum().item()
total = eval(template_vars["LABELS_VAR"]).size(0)
{NODE_ID} = correct / total if total > 0 else 0.0

print(f"Accuracy: {{NODE_ID}} = {{{NODE_ID}:.2%}}")
