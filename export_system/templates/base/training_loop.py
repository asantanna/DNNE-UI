# Generic training loop template
def train(context, dataloader, num_epochs=10):
    """Main training loop"""
    
    # Collect parameters from context
    params = []
    for key, value in context.memory.items():
        if isinstance(value, nn.Module):
            params.extend(value.parameters())
    
    # Create optimizer
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Forward pass
            # [NODE EXECUTION CODE WILL BE INSERTED HERE]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            context.step_count += 1
        
        context.episode_count += 1
        print(f"Epoch {epoch + 1}/{num_epochs} complete")
