#!/usr/bin/env python3
"""
Debug script with download progress for MNIST
"""
print("foo")
import torch
from torchvision import datasets, transforms
import asyncio
import time
import os

async def test_mnist():
    print("Testing MNIST dataset loading...")
    
    # Check if data already exists
    data_path = "./data"
    mnist_path = os.path.join(data_path, "MNIST", "raw")
    
    if os.path.exists(mnist_path) and len(os.listdir(mnist_path)) > 0:
        print("✅ MNIST data already exists, no download needed")
    else:
        print("⏳ MNIST data not found, will download (~60MB)")
        print("   This may take a few minutes on first run...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("\nCreating MNIST dataset...")
    start_time = time.time()
    
    # Create dataset with download
    dataset = datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )
    
    elapsed = time.time() - start_time
    print(f"✅ Dataset ready in {elapsed:.1f}s: {len(dataset)} samples")
    
    print("\nCreating DataLoader...")
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    print("✅ DataLoader created")
    
    print("\nGetting first batch...")
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    print(f"✅ Got batch: images shape={images.shape}, labels shape={labels.shape}")
    print(f"   Image stats: min={images.min():.4f}, max={images.max():.4f}, mean={images.mean():.4f}")
    
    print("\nSimulating sensor node at 2Hz...")
    for i in range(5):
        start = time.time()
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, labels = next(data_iter)
            print("   (Restarted epoch)")
        
        elapsed = time.time() - start
        print(f"Batch {i+1}: Got data in {elapsed:.4f}s, shape={images.shape}")
        await asyncio.sleep(0.5)  # 2Hz rate
    
    print("\n✅ Test complete!")
    print("\nNow you can run the generated queue scripts.")
    print("They should start producing output immediately.")

if __name__ == "__main__":
    print("Starting MNIST test script...")
    asyncio.run(test_mnist())
    
