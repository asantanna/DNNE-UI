#!/usr/bin/env python3
"""
Generate test CSV data for Franka robot arm in task-space (6 DOF)
Creates smooth sinusoidal trajectories for testing with FrankaCubeStack
"""

import numpy as np
import pandas as pd
import json

def generate_franka_trajectory(num_samples=1000, frequency_hz=100.0):
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
    columns = ["x", "y", "z", "rx", "ry", "rz"]
    df = pd.DataFrame(np.array(trajectories).T, columns=columns)
    
    # Add timestamp column
    df.insert(0, 'timestamp', t)
    
    return df

def create_metadata(frequency_hz=100.0, num_samples=1000):
    """Create metadata JSON for the trajectory data"""
    
    metadata = {
        "file_id": "franka_taskspace_trajectory_v1",
        "description": "Test task-space trajectories for Franka robot arm (FrankaCubeStack)",
        "robot": "Franka Emika Panda",
        "control_mode": "task_space",
        "columns": [
            {"name": "timestamp", "dtype": "float64", "unit": "seconds"},
            {"name": "x", "dtype": "float32", "unit": "meters", "description": "X position delta"},
            {"name": "y", "dtype": "float32", "unit": "meters", "description": "Y position delta"},
            {"name": "z", "dtype": "float32", "unit": "meters", "description": "Z position delta"},
            {"name": "rx", "dtype": "float32", "unit": "radians", "description": "X rotation delta"},
            {"name": "ry", "dtype": "float32", "unit": "radians", "description": "Y rotation delta"},
            {"name": "rz", "dtype": "float32", "unit": "radians", "description": "Z rotation delta"},
        ],
        "frequency_hz": frequency_hz,
        "total_rows": num_samples,
        "duration_seconds": num_samples / frequency_hz,
        "created": "2025-01-12T10:00:00Z",
        "version": "1.0.0"
    }
    
    return metadata

def main():
    """Generate test data files"""
    
    # Parameters
    num_samples = 1000
    frequency_hz = 100.0
    
    # Generate trajectory data
    print("Generating Franka trajectory data...")
    df = generate_franka_trajectory(num_samples, frequency_hz)
    
    # Save CSV
    csv_path = "test_data/franka_trajectory.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved trajectory data to {csv_path}")
    
    # Create and save metadata
    metadata = create_metadata(frequency_hz, num_samples)
    metadata_path = "test_data/franka_trajectory_metadata.json"
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    # Print sample of data
    print("\nFirst 5 rows of data:")
    print(df.head())
    
    print("\nData statistics:")
    print(df.describe())
    
    print(f"\nGenerated {num_samples} samples at {frequency_hz} Hz")
    print(f"Total duration: {num_samples/frequency_hz:.1f} seconds")

if __name__ == "__main__":
    main()