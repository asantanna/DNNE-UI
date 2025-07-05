#!/usr/bin/env python3
"""
Direct PyTorch MNIST training for performance comparison
Same network architecture and hyperparameters as DNNE export
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleNetwork(nn.Module):
    """Same network as DNNE: 784 -> 128 -> 128 -> 10"""
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        # Flatten if needed (for MNIST)
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        return self.network(x)

def main():
    print("🚀 Direct PyTorch MNIST Training Benchmark")
    print("=" * 60)
    
    # Same parameters as DNNE export
    batch_size = 32
    learning_rate = 0.1
    momentum = 0.9
    num_epochs = 1
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load MNIST data (same as DNNE)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataset loaded: {len(train_dataset)} samples, {len(train_loader)} batches")
    
    # Create model, loss, optimizer (same as DNNE)
    model = SimpleNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Optimizer: SGD(lr={learning_rate}, momentum={momentum})")
    print("=" * 60)
    
    # Training loop
    start_time = time.time()
    model.train()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Starting Epoch {epoch}")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start = time.time()
            
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            batch_time = time.time() - batch_start
            
            # Log every 10 batches (same as DNNE verbose mode)
            if batch_idx % 10 == 0:
                accuracy = 100. * correct / total
                avg_loss = running_loss / (batch_idx + 1)
                print(f"Batch {batch_idx:4d}/{len(train_loader)} ({100.*batch_idx/len(train_loader):5.1f}%) "
                      f"- Loss: {loss.item():.4f}, Acc: {accuracy:5.2f}%, "
                      f"Time: {batch_time*1000:.1f}ms")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        
        print("=" * 60)
        print(f"📊 EPOCH {epoch} COMPLETE")
        print(f"   Time: {epoch_time:.2f}s")
        print(f"   Batches: {len(train_loader)}")
        print(f"   Avg Loss: {epoch_loss:.4f}")
        print(f"   Avg Accuracy: {epoch_accuracy:.2f}%")
        print(f"   Throughput: {len(train_loader)/epoch_time:.1f} batches/sec")
        print("=" * 60)
    
    total_time = time.time() - start_time
    print(f"\n🎯 TRAINING COMPLETE!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average batch time: {total_time/len(train_loader)*1000:.1f}ms")
    print(f"Overall throughput: {len(train_loader)/total_time:.1f} batches/sec")

if __name__ == "__main__":
    main()