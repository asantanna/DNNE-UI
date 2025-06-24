# Template variables - replaced during export
template_vars = {
    "NODE_ID": "loss_1",
    "PREDICTIONS_VAR": "predictions",
    "LABELS_VAR": "labels"
}

# Cross Entropy Loss: {NODE_ID}
{NODE_ID} = F.cross_entropy(
    eval(template_vars["PREDICTIONS_VAR"]),
    eval(template_vars["LABELS_VAR"])
)

{NODE_ID}_value = {NODE_ID}.item()
