# Template variables - replaced during export
template_vars = {
    "NODE_ID": "node_1",
    "DATA_PATH": "./data",
    "TRAIN": True,
    "DOWNLOAD": True
}

# Extract variables for cleaner code
NODE_ID = template_vars["NODE_ID"]
DATA_PATH = template_vars["DATA_PATH"]
TRAIN = template_vars["TRAIN"]
DOWNLOAD = template_vars["DOWNLOAD"]

# MNIST Dataset
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST(
    root=DATA_PATH,
    train=TRAIN,
    download=DOWNLOAD,
    transform=transform
)

# Make dataset available with node's name
globals()[NODE_ID] = dataset
globals()[f"{NODE_ID}_output"] = dataset

print(f"Loaded MNIST dataset '{NODE_ID}': {len(dataset)} samples")