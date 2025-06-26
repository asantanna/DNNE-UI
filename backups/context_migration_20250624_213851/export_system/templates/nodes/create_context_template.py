# Template variables - replaced during export
template_vars = {
    "NODE_ID": "context_1",
    "TRAINING_MODE": True
}

# Create Context: {NODE_ID}
{NODE_ID} = Context()
{NODE_ID}.training = template_vars["TRAINING_MODE"]
