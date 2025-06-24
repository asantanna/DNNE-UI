# Template variables - replaced during export
template_vars = {
    "NODE_ID": "node_2",
    "DATASET_VAR": "node_1_output",
    "BATCH_SIZE": 32,
    "SHUFFLE": True,
    "NUM_WORKERS": 0
}

# Extract variables
NODE_ID = template_vars["NODE_ID"]
DATASET_VAR = template_vars["DATASET_VAR"]
BATCH_SIZE = template_vars["BATCH_SIZE"]
SHUFFLE = template_vars["SHUFFLE"]
NUM_WORKERS = template_vars["NUM_WORKERS"]

# Batch Sampler (DataLoader)
from torch.utils.data import DataLoader

# Get the dataset from the variable name
dataset = globals()[DATASET_VAR]

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS
)

# Make dataloader available with node's name
globals()[NODE_ID] = dataloader
globals()[f"{NODE_ID}_output"] = dataloader

print(f"Created DataLoader '{NODE_ID}': batch_size={BATCH_SIZE}, shuffle={SHUFFLE}")
