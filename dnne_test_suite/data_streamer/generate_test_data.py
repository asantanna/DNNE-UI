#!/usr/bin/env python3
"""
Generate test CSV data for Franka robot arm in task-space (6 DOF)
Creates smooth sinusoidal trajectories for testing with Isaac Gym environments
"""

import numpy as np
import pandas as pd
import json
import argparse
import sys

# Environment configurations with their timesteps
ENV_CONFIGS = {
    "FrankaCubeStack": {
        "dt": 0.01667,  # 60 Hz
        "description": "Franka cube stacking task"
    },
    "FrankaCabinet": {
        "dt": 0.01667,  # 60 Hz  
        "description": "Franka cabinet opening task"
    },
    "Cartpole": {
        "dt": 0.01667,  # 60 Hz
        "description": "Classic cartpole balance task"
    },
    "Ant": {
        "dt": 0.01667,  # 60 Hz
        "description": "Ant locomotion task"
    },
    "Humanoid": {
        "dt": 0.0083,   # 120 Hz
        "description": "Humanoid locomotion task"
    },
    "Shadow": {
        "dt": 0.01667,  # 60 Hz
        "description": "Shadow hand manipulation"
    },
    # Add more environments as needed
}

def generate_franka_trajectory(num_samples=1000, frequency_hz=60.0):
    """Generate smooth sinusoidal trajectories for Franka's task-space control (6 DOF)"""
    
    # Time array
    duration = num_samples / frequency_hz  # seconds
    t = np.linspace(0, duration, num_samples)
    
    # Task-space limits for FrankaCubeStack
    # Based on the cmd_limit values in franka_cube_stack.py:
    # [0.1, 0.1, 0.1, 0.5, 0.5, 0.5] for [x, y, z, rx, ry, rz]
    # These are action scale limits, we'll use smaller values for safety
    task_limits = [
        (-0.05, 0.05),   # X position
        (-0.05, 0.05),   # Y position  
        (-0.05, 0.05),   # Z position
        (-0.25, 0.25),   # X rotation
        (-0.25, 0.25),   # Y rotation
        (-0.25, 0.25),   # Z rotation
        (-1.0, 1.0),     # Gripper (>= 0 open, < 0 closed)
    ]
    
    # Generate smooth trajectories for each DOF
    # Use different frequencies and phases for variety
    trajectories = []
    
    for i, (min_val, max_val) in enumerate(task_limits):
        # Center and amplitude
        center = 0.0  # Task-space commands are typically centered at 0
        amplitude = (max_val - min_val) * 0.4  # Use 40% of range for safety
        
        # Different frequency for each DOF (0.1 to 0.4 Hz)
        freq = 0.1 + (i * 0.05)
        
        # Phase offset for each DOF
        phase = i * np.pi / 6
        
        # Generate sinusoidal trajectory
        trajectory = center + amplitude * np.sin(2 * np.pi * freq * t + phase)
        trajectories.append(trajectory)
    
    # Create DataFrame with task-space column names
    columns = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    df = pd.DataFrame(np.array(trajectories).T, columns=columns)
    
    # Don't add timestamp - Isaac Gym expects just the action values
    return df

def create_metadata(frequency_hz=100.0, num_samples=1000):
    """Create metadata JSON for the trajectory data"""
    
    metadata = {
        "file_id": "franka_taskspace_trajectory_v2",
        "description": "Test task-space trajectories for Franka robot arm with gripper (FrankaCubeStack)",
        "robot": "Franka Emika Panda",
        "control_mode": "task_space_with_gripper",
        "columns": [
            {"name": "x", "dtype": "float32", "unit": "meters", "description": "X position delta"},
            {"name": "y", "dtype": "float32", "unit": "meters", "description": "Y position delta"},
            {"name": "z", "dtype": "float32", "unit": "meters", "description": "Z position delta"},
            {"name": "rx", "dtype": "float32", "unit": "radians", "description": "X rotation delta"},
            {"name": "ry", "dtype": "float32", "unit": "radians", "description": "Y rotation delta"},
            {"name": "rz", "dtype": "float32", "unit": "radians", "description": "Z rotation delta"},
            {"name": "gripper", "dtype": "float32", "unit": "normalized", "description": "Gripper command (>=0 open, <0 closed)"},
        ],
        "frequency_hz": frequency_hz,
        "total_rows": num_samples,
        "duration_seconds": num_samples / frequency_hz,
        "created": "2025-01-12T10:00:00Z",
        "version": "1.0.0"
    }
    
    return metadata

def get_output_filename(env_name):
    """Generate output filename based on environment name"""
    # Convert CamelCase to snake_case for filename
    import re
    snake_case = re.sub('([A-Z]+)', r'_\1', env_name).lower().strip('_')
    
    # Special cases for cleaner names
    name_map = {
        "franka_cube_stack": "franka_cubes",
        "franka_cabinet": "franka_cabinet", 
        "cart_pole": "cartpole",
        "shadow_hand": "shadow_hand",
    }
    
    base_name = name_map.get(snake_case, snake_case)
    return f"{base_name}_test.csv"

def main():
    """Generate test data files with command-line parameters"""
    parser = argparse.ArgumentParser(description='Generate sinusoidal test data for Isaac Gym environments')
    parser.add_argument('--env', type=str, default='FrankaCubeStack',
                        help='Environment name (e.g., FrankaCubeStack, FrankaCabinet, Cartpole)')
    parser.add_argument('--seconds', type=float, default=5.0,
                        help='Duration of trajectory in seconds (default: 5.0)')
    parser.add_argument('--list-envs', action='store_true',
                        help='List available environments and exit')
    
    args = parser.parse_args()
    
    # List environments if requested
    if args.list_envs:
        print("\nAvailable environments:")
        for env_name, config in ENV_CONFIGS.items():
            freq = 1.0 / config['dt']
            print(f"  {env_name:20} - {config['description']} ({freq:.0f} Hz)")
        sys.exit(0)
    
    # Get environment configuration
    if args.env not in ENV_CONFIGS:
        print(f"Error: Unknown environment '{args.env}'")
        print(f"Available environments: {', '.join(ENV_CONFIGS.keys())}")
        sys.exit(1)
    
    env_config = ENV_CONFIGS[args.env]
    dt = env_config['dt']
    frequency_hz = 1.0 / dt
    
    # Calculate number of samples
    num_samples = int(args.seconds * frequency_hz)
    
    print(f"\n=== Generating test data for {args.env} ===")
    print(f"Duration: {args.seconds} seconds")
    print(f"Frequency: {frequency_hz:.0f} Hz (dt={dt})")
    print(f"Samples: {num_samples}")
    
    # Generate sinusoidal test data
    print(f"\nGenerating sinusoidal trajectory data...")
    df = generate_franka_trajectory(num_samples, frequency_hz)
    
    # Generate output filename
    output_filename = get_output_filename(args.env)
    csv_path = f"test_data/{output_filename}"
    
    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to: {csv_path}")
    
    # Create and save metadata
    metadata = create_metadata(frequency_hz, num_samples)
    metadata['environment'] = args.env
    metadata['pattern'] = 'sinusoidal'
    metadata_path = csv_path.replace('.csv', '_metadata.json')
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")
    
    # Print sample of data
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nLast 5 rows:")
    print(df.tail())
    
    print(f"\nSuccessfully generated {num_samples} samples ({args.seconds} seconds at {frequency_hz:.0f} Hz)")

if __name__ == "__main__":
    main()