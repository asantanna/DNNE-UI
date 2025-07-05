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
import argparse

class SimpleNetwork(nn.Module):
    """Improved network with batch normalization and dropout for stability"""
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        # Flatten if needed (for MNIST)
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        return self.network(x)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MNIST Training Benchmark')
    parser.add_argument('--disable_scheduler', action='store_true', 
                        help='Disable learning rate scheduler')
    args = parser.parse_args()
    
    print("🚀 Direct PyTorch MNIST Training Benchmark")
    print("=" * 60)
    
    # Improved parameters for stability based on web best practices
    batch_size = 64  # Increased from 32 for better stability
    learning_rate = 0.01  # Reduced from 0.1 for more stable training
    momentum = 0.9
    num_epochs = 10  # Full training for proper convergence
    use_scheduler = not args.disable_scheduler
    
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
    
    # Create model, loss, optimizer with optional learning rate scheduler
    model = SimpleNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) if use_scheduler else None
    
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Optimizer: SGD(lr={learning_rate}, momentum={momentum})")
    print(f"Scheduler: {'StepLR(step_size=7, gamma=0.1)' if use_scheduler else 'Disabled'}")
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
            
            # Log every 100 batches to reduce output with longer training
            if batch_idx % 100 == 0:
                accuracy = 100. * correct / total
                avg_loss = running_loss / (batch_idx + 1)
                current_lr = optimizer.param_groups[0]['lr']
                lr_info = f"LR: {current_lr:.6f}, " if use_scheduler else ""
                print(f"Batch {batch_idx:4d}/{len(train_loader)} ({100.*batch_idx/len(train_loader):5.1f}%) "
                      f"- Loss: {loss.item():.4f}, Acc: {accuracy:5.2f}%, "
                      f"{lr_info}Time: {batch_time*1000:.1f}ms")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"📊 EPOCH {epoch} COMPLETE")
        print(f"   Time: {epoch_time:.2f}s")
        print(f"   Batches: {len(train_loader)}")
        print(f"   Avg Loss: {epoch_loss:.4f}")
        print(f"   Avg Accuracy: {epoch_accuracy:.2f}%")
        if use_scheduler:
            print(f"   Learning Rate: {current_lr:.6f}")
        print(f"   Throughput: {len(train_loader)/epoch_time:.1f} batches/sec")
        print("=" * 60)
        
        # Update learning rate scheduler if enabled
        if scheduler:
            scheduler.step()
    
    total_time = time.time() - start_time
    print(f"\n🎯 TRAINING COMPLETE!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average epoch time: {total_time/num_epochs:.2f}s")
    print(f"Overall throughput: {len(train_loader)*num_epochs/total_time:.1f} batches/sec")
    
    # Test the model
    print("\n📊 TESTING MODEL...")
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / total
    
    print("=" * 60)
    print(f"📊 TEST RESULTS")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.2f}%")
    print(f"   Samples: {total}")
    print("=" * 60)

if __name__ == "__main__":
    main()