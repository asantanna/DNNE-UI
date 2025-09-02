#!/usr/bin/env python3
"""
Direct test: Can we learn Isaac Gym physics?
Tests if a network can learn (obs(t), action(t)) -> obs(t+1) for FrankaDNNE.

Usage:
    python shadow_train_standalone.py [--timeout SECONDS] [--lr LEARNING_RATE]
"""

import sys
import numpy as np
from pathlib import Path
import csv
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add paths
sys.path.append('/home/asantanna/DNNE/DNNE-LINUX-SUPPORT/isaacgym/python')
sys.path.append('/home/asantanna/DNNE/DNNE-LINUX-SUPPORT/IsaacGymEnvs')

# Import Isaac Gym FIRST
from isaacgym import gymapi, gymutil, gymtorch
from isaacgymenvs.tasks.franka_dnne import FrankaDNNE

# Now import torch
import torch
import torch.nn as nn
import torch.optim as optim


class PhysicsNet(nn.Module):
    """27 inputs (7 actions + 20 obs) -> 20 outputs (next obs)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(27, 128),
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 20)
        )
    
    def forward(self, x):
        return self.net(x)


def main(args):
    print("="*60, flush=True)
    print("CAN WE LEARN ISAAC GYM PHYSICS?", flush=True)
    print("="*60, flush=True)
    print(f"Timeout: {args.timeout}s, Learning rate: {args.lr}", flush=True)
    
    device = "cuda:0"
    start_time = time.time()
    
    # Load CSV actions for diversity
    print("Loading CSV actions...", flush=True)
    csv_path = "/home/asantanna/DNNE/DNNE-UI/dnne_test_suite/data_streamer/test_data/franka_sinusoidal_exploration_001.csv"
    csv_data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_data.append([float(row[f'torque_joint{i}']) for i in range(7)])
    csv_actions = torch.tensor(csv_data, dtype=torch.float32, device=device)
    print(f"Loaded {len(csv_actions)} CSV actions", flush=True)
    
    # Create FrankaDNNE environment directly
    print("\nCreating FrankaDNNE environment...", flush=True)
    
    # Full config with all required fields
    cfg = {
        "physics_engine": "physx",
        "env": {
            "numEnvs": 1,
            "envSpacing": 2.0,
            "episodeLength": 1000,
            "resetDist": 0.0,
            "actionScale": 7.0,
            "startPositionNoise": 0.0,
            "startRotationNoise": 0.0,
            "frankaPositionNoise": 0.0,
            "frankaRotationNoise": 0.0,
            "frankaDofNoise": 0.0,
            "aggregateMode": 0,
            "distRewardScale": 0.0,
            "liftRewardScale": 0.0,
            "alignRewardScale": 0.0,
            "stackRewardScale": 0.0,
            "controlType": "joint_tor",
            "enableDebugVis": False,
            "dofVelocityScale": 0.1,
            "controlFrequencyInv": 1,
            "observation_mode": "random_target",
            "control_mode": "joint_tor",
            "numActions": 7,
            "numObservations": 20,
            "randomize": False,
            "randomization_params": {
                "frequency": 600,
                "observations": {"range": [0, 0.002], "operation": "additive_gaussian"},
                "actions": {"range": [0.0, 0.02], "operation": "additive_gaussian"}
            }
        },
        "sim": {
            "dt": 1.0/60.0,
            "substeps": 2,
            "up_axis": "z",
            "use_gpu_pipeline": True,
            "gravity": [0.0, 0.0, -9.81],
            "physx": {
                "num_threads": 0,
                "solver_type": 1,
                "use_gpu": True,
                "num_position_iterations": 4,
                "num_velocity_iterations": 1,
                "contact_offset": 0.002,
                "rest_offset": 0.001,
                "bounce_threshold_velocity": 0.2,
                "max_depenetration_velocity": 100.0,
                "default_buffer_size_multiplier": 5.0,
                "max_gpu_contact_pairs": 8388608,
                "num_subscenes": 0,
                "contact_collection": 0
            }
        },
        "task": {
            "randomize": False
        }
    }
    
    # Create environment with all required arguments
    print("Calling FrankaDNNE constructor...", flush=True)
    env = FrankaDNNE(
        cfg=cfg,
        sim_device=device,
        graphics_device_id=0,
        headless=True,
        rl_device=device,
        virtual_screen_capture=False,
        force_render=False
    )
    
    print(f"Environment ready: obs_dim={env.num_obs}, act_dim={env.num_acts}", flush=True)
    
    # Create network and optimizer
    net = PhysicsNet().to(device)
    opt = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    
    # Training
    train_loss_arg = args.train_loss if hasattr(args, 'train_loss') else args.__dict__.get('train-loss', 'shadow')
    mix_ratio = args.loss_mix_ratio if hasattr(args, 'loss_mix_ratio') else args.__dict__.get('loss-mix-ratio', 0.8)
    if train_loss_arg == 'shadow':
        print(f"\nTraining to learn physics with {train_loss_arg} loss (mix={mix_ratio:.2f})...", flush=True)
    else:
        print(f"\nTraining to learn physics with {train_loss_arg} loss...", flush=True)
    print("-"*60, flush=True)
    
    # Track all loss types
    shadow_losses = []
    position_losses = []
    joint_losses = []
    full_losses = []
    print("Calling env.reset()...", flush=True)
    obs_dict = env.reset()
    print(f"Reset returned, obs_dict keys: {obs_dict.keys() if isinstance(obs_dict, dict) else 'Not a dict!'}", flush=True)
    
    # Check if reset returns a dict or tensor
    if isinstance(obs_dict, dict):
        obs = obs_dict.get("obs", obs_dict)
    else:
        obs = obs_dict
    print(f"Initial obs shape: {obs.shape if hasattr(obs, 'shape') else 'No shape!'}", flush=True)
    
    # Step with null action to get first real observation (like in workflow)
    null_action = torch.zeros(1, 7, device=device)
    print("Stepping with null action...", flush=True)
    obs_dict, _, _, _ = env.step(null_action)
    if isinstance(obs_dict, dict):
        obs = obs_dict["obs"]
    else:
        obs = obs_dict
    print(f"After null step, obs shape: {obs.shape}", flush=True)
    
    csv_idx = 0
    
    print("Starting training loop...", flush=True)
    step = 0
    while True:
        # Check timeout
        if time.time() - start_time > args.timeout:
            print(f"\nTimeout reached ({args.timeout}s)", flush=True)
            break
            
        # Use ONLY CSV actions, exactly like the workflow
        action = csv_actions[csv_idx % len(csv_actions)].unsqueeze(0)
        csv_idx += 1
        
        # Current state
        obs_t = obs
        
        # Step physics
        obs_dict_t1, _, dones, _ = env.step(action)
        obs_t1 = obs_dict_t1["obs"]
        
        # Learn mapping
        x = torch.cat([action, obs_t], dim=1)
        pred_obs_t1 = net(x)
        
        # Compute basic errors
        # EEF error
        actual_eef = obs_t1[:, 3:6]       # Actual EEF position from simulator
        pred_eef = pred_obs_t1[:, 3:6]    # Predicted EEF position
        eef_error = torch.norm(pred_eef - actual_eef, p=2, dim=1)
        
        # Joint error
        actual_joints = obs_t1[:, 10:19]   # 9 joint angles
        pred_joints = pred_obs_t1[:, 10:19] # Predicted joint angles
        joint_error = torch.norm(pred_joints - actual_joints, p=2, dim=1)
        joint_error_normalized = joint_error / 3.14159
        
        # Define losses based on the errors
        position_loss = eef_error.mean()  # "position" means EEF error
        joint_loss = joint_error_normalized.mean()  # "joint" means joint error
        
        # Shadow loss is a mix of EEF and joint errors
        mix_ratio = args.loss_mix_ratio if hasattr(args, 'loss_mix_ratio') else args.__dict__.get('loss-mix-ratio', 0.8)
        shadow_loss = (mix_ratio * eef_error + (1 - mix_ratio) * joint_error_normalized).mean()
        
        # Full loss (all 20 dims)
        full_loss = torch.norm(pred_obs_t1 - obs_t1, p=2, dim=1).mean()
        
        # Select which loss to use for training
        train_loss_arg = args.train_loss if hasattr(args, 'train_loss') else args.__dict__.get('train-loss', 'shadow')
        if train_loss_arg == "shadow":
            train_loss = shadow_loss
        elif train_loss_arg == "position":
            train_loss = position_loss
        elif train_loss_arg == "joint":
            train_loss = joint_loss
        else:  # full
            train_loss = full_loss
        
        # Backprop with selected loss
        opt.zero_grad()
        train_loss.backward()
        grad_norm = torch.norm(torch.cat([p.grad.flatten() for p in net.parameters()]))
        opt.step()
        
        # Track all losses
        shadow_losses.append(shadow_loss.item())
        position_losses.append(position_loss.item())
        joint_losses.append(joint_loss.item())
        full_losses.append(full_loss.item())
        
        
        # Print progress
        if step % 100 == 0:
            # Get average of whatever we're training with
            train_loss_arg = args.train_loss if hasattr(args, 'train_loss') else args.__dict__.get('train-loss', 'shadow')
            if train_loss_arg == "shadow":
                train_losses_list = shadow_losses
            elif train_loss_arg == "position":
                train_losses_list = position_losses
            elif train_loss_arg == "joint":
                train_losses_list = joint_losses
            else:
                train_losses_list = full_losses
            recent = np.mean(train_losses_list[-100:]) if len(train_losses_list) >= 100 else train_loss.item()
            
            print(f"Step {step:4d}: Train={train_loss.item():.4f}, "
                  f"Avg100={recent:.4f}, Shadow={shadow_loss.item():.4f}, "
                  f"Joint={joint_loss.item():.4f}, Grad={grad_norm.item():.3f}")
            
            # Show prediction vs actual (EEF and joints)
            if step % 500 == 0:
                with torch.no_grad():
                    print(f"  Actual EEF:    {actual_eef[0].cpu().numpy()}")
                    print(f"  Predicted EEF: {pred_eef[0].cpu().numpy()}")
                    print(f"  EEF error:     {eef_error[0].item():.4f}")
                    print(f"  Joint error:   {joint_error[0].item():.4f} (norm: {joint_error_normalized[0].item():.4f})")
        
        # Reset if done
        if dones.any():
            obs_dict = env.reset()
            # Step with null action after reset
            null_action = torch.zeros(1, 7, device=device)
            obs_dict, _, _, _ = env.step(null_action)
            obs = obs_dict["obs"]
            csv_idx = 0
        else:
            obs = obs_t1
        
        step += 1
    
    # Analysis
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    
    # Analyze the loss we trained with
    train_loss_arg = args.train_loss if hasattr(args, 'train_loss') else args.__dict__.get('train-loss', 'shadow')
    if train_loss_arg == "shadow":
        train_losses_list = shadow_losses
        loss_name = "Shadow_Train loss"
    elif train_loss_arg == "position":
        train_losses_list = position_losses
        loss_name = "Position loss"
    elif train_loss_arg == "joint":
        train_losses_list = joint_losses
        loss_name = "Joint loss"
    else:
        train_losses_list = full_losses
        loss_name = "Full loss"
    
    initial = np.mean(train_losses_list[:100]) if len(train_losses_list) >= 100 else train_losses_list[0]
    final = np.mean(train_losses_list[-100:]) if len(train_losses_list) >= 100 else train_losses_list[-1]
    improvement = initial - final
    pct = (improvement / initial) * 100
    
    print(f"Training with: {loss_name}")
    print(f"Initial loss: {initial:.6f}")
    print(f"Final loss:   {final:.6f}")
    print(f"Improvement:  {improvement:.6f} ({pct:.1f}%)")
    
    if pct > 10:
        print("\n✅ SUCCESS: We CAN learn Isaac Gym physics!")
    else:
        print("\n❌ FAILED: Cannot learn physics with current setup")
        print("Try: Higher learning rate, more diverse actions, longer training")
    
    # env.close() # FrankaDNNE doesn't have close method
    # Store CSV length for milestone calculation
    main.csv_length = len(csv_actions)
    # Return all losses
    return {
        'shadow': shadow_losses,
        'position': position_losses,
        'joint': joint_losses,
        'full': full_losses
    }


def plot_losses(all_losses, lr, timeout, train_loss, plot_losses, milestone_interval=None, csv_length=1000, mix_ratio=0.8):
    """Plot the training losses
    
    Args:
        all_losses: dict with keys 'shadow', 'position', 'joint', 'full'
        lr: learning rate used
        timeout: timeout value used
        train_loss: which loss was used for training
        plot_losses: list of loss names to plot
        milestone_interval: interval for milestones
        csv_length: length of CSV for default milestone calculation
        mix_ratio: mixing ratio for shadow loss
    """
    # Prepare data for plotting
    loss_colors = {
        'shadow': 'blue',
        'position': 'green', 
        'joint': 'orange',
        'full': 'red'
    }
    
    loss_labels = {
        'shadow': f'Shadow ({int(mix_ratio*100)}% EEF + {int((1-mix_ratio)*100)}% Joint)',
        'position': 'Position (EEF error)',
        'joint': 'Joint angles',
        'full': 'Full (all 20 dims)'
    }
    
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Raw losses with moving average
    plt.subplot(1, 2, 1)
    
    for loss_name in plot_losses:
        if loss_name in all_losses:
            loss_data = all_losses[loss_name]
            color = loss_colors.get(loss_name, 'black')
            label = loss_labels.get(loss_name, loss_name)
            
            # Plot raw data with transparency
            plt.plot(loss_data, color=color, alpha=0.3, linewidth=0.5)
            
            # Add moving average
            if len(loss_data) > 100:
                window = 100
                moving_avg = np.convolve(loss_data, np.ones(window)/window, mode='valid')
                plt.plot(range(window-1, len(loss_data)), moving_avg, 
                        color=color, label=label, linewidth=2)
            else:
                plt.plot(loss_data, color=color, label=label)
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    if train_loss == 'shadow':
        plt.title(f'Training Progress (lr={lr}, train={train_loss}, mix={mix_ratio:.2f})')
    else:
        plt.title(f'Training Progress (lr={lr}, train={train_loss})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Milestones for all requested losses
    plt.subplot(1, 2, 2)
    
    # Use provided milestone interval or default to 1/4 of CSV length
    if milestone_interval is None:
        milestone_interval = csv_length // 4
    
    for loss_name in plot_losses:
        if loss_name in all_losses:
            loss_data = all_losses[loss_name]
            color = loss_colors.get(loss_name, 'black')
            label = loss_labels.get(loss_name, loss_name)
            
            if len(loss_data) > milestone_interval:
                milestones = list(range(0, len(loss_data), milestone_interval))
                milestone_losses = [np.mean(loss_data[max(0,i-50):min(len(loss_data),i+50)]) 
                                  for i in milestones]
                plt.plot(milestones, milestone_losses, 'o-', 
                        color=color, markersize=6, label=label)
    
    # Add improvement text for training loss
    if train_loss in all_losses and len(all_losses[train_loss]) > milestone_interval:
        loss_data = all_losses[train_loss]
        milestones = list(range(0, len(loss_data), milestone_interval))
        milestone_losses = [np.mean(loss_data[max(0,i-50):min(len(loss_data),i+50)]) 
                          for i in milestones]
        if len(milestone_losses) > 1:
            initial = milestone_losses[0]
            final = milestone_losses[-1]
            improvement = (initial - final) / initial * 100
            plt.text(0.5, 0.95, f'{train_loss} improvement: {improvement:.1f}%', 
                    transform=plt.gca().transAxes, 
                    ha='center', va='top', fontsize=11, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.xlabel('Step')
    plt.ylabel('Loss (averaged)')
    plt.title(f'Loss at Milestones (every {milestone_interval} steps)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('isaac_physics_losses.png', dpi=100)
    print(f"\nPlot saved to isaac_physics_losses.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test learning Isaac Gym physics")
    parser.add_argument("--timeout", type=float, default=30.0, 
                        help="Timeout in seconds (default: 30)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--train-loss", type=str, default="shadow", 
                        choices=["shadow", "position", "joint", "full"],
                        help="Loss function for training: shadow (80%% EEF + 20%% joints), position, joint, or full")
    parser.add_argument("--plot", type=str, default="train",
                        help="Comma-separated list of losses to plot: shadow,position,joint,full or 'train' for training loss")
    parser.add_argument("--milestone", type=int, default=None,
                        help="Milestone interval for loss plot (default: 1/4 of CSV length)")
    parser.add_argument("--loss-mix-ratio", type=float, default=0.8,
                        help="Mix ratio for shadow loss: mix_val * EEF + (1-mix_val) * joints (default: 0.8)")
    
    args = parser.parse_args()
    
    # Run training
    all_losses = main(args)
    
    # Save all losses
    for loss_name, loss_data in all_losses.items():
        filename = f"isaac_physics_{loss_name}_losses.npy"
        np.save(filename, loss_data)
        print(f"Saved {len(loss_data)} {loss_name} loss values to {filename}")
    
    # Parse plot argument
    plot_arg = args.plot
    train_loss_arg = args.train_loss if hasattr(args, 'train_loss') else args.__dict__.get('train-loss', 'shadow')
    
    if plot_arg == 'train':
        # plot only the training loss
        losses_to_plot = [train_loss_arg]
    else:
        # Parse comma-separated list
        losses_to_plot = [l.strip() for l in plot_arg.split(',')]
    
    # Plot losses if we have data
    if all_losses and any(len(all_losses.get(l, [])) > 0 for l in losses_to_plot):
        csv_length = getattr(main, 'csv_length', 1000)
        mix_ratio = args.loss_mix_ratio if hasattr(args, 'loss_mix_ratio') else args.__dict__.get('loss-mix-ratio', 0.8)
        plot_losses(all_losses, args.lr, args.timeout, 
                   train_loss_arg, losses_to_plot,
                   milestone_interval=args.milestone,
                   csv_length=csv_length,
                   mix_ratio=mix_ratio)