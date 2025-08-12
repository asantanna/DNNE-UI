#!/usr/bin/env python3
"""
Generate test CSV data for Franka robot arm with 7 DOF
Creates smooth sinusoidal trajectories for testing
"""

import numpy as np
import pandas as pd
import json

def generate_franka_trajectory(num_samples=1000, frequency_hz=100.0):
    """Generate smooth sinusoidal trajectories for Franka's 7 joints"""
    
    # Time array
    duration = num_samples / frequency_hz  # seconds
    t = np.linspace(0, duration, num_samples)
    
    # Joint limits for Franka (approximate, in radians)
    # These are conservative limits for safe testing
    joint_limits = [
        (-2.8, 2.8),   # Joint 0
        (-1.7, 1.7),   # Joint 1  
        (-2.8, 2.8),   # Joint 2
        (-3.0, -0.06), # Joint 3
        (-2.8, 2.8),   # Joint 4
        (-0.01, 3.75), # Joint 5
        (-2.8, 2.8),   # Joint 6
    ]
    
    # Generate smooth trajectories for each joint
    # Use different frequencies and phases for variety
    trajectories = []
    
    for i, (min_val, max_val) in enumerate(joint_limits):
        # Center and amplitude
        center = (min_val + max_val) / 2
        amplitude = (max_val - min_val) * 0.3  # Use 30% of range for safety
        
        # Different frequency for each joint (0.1 to 0.5 Hz)
        freq = 0.1 + (i * 0.05)
        
        # Phase offset for each joint
        phase = i * np.pi / 7
        
        # Generate sinusoidal trajectory
        trajectory = center + amplitude * np.sin(2 * np.pi * freq * t + phase)
        trajectories.append(trajectory)
    
    # Create DataFrame
    columns = [f"joint_{i}" for i in range(7)]
    df = pd.DataFrame(np.array(trajectories).T, columns=columns)
    
    # Add timestamp column
    df.insert(0, 'timestamp', t)
    
    return df

def create_metadata(frequency_hz=100.0, num_samples=1000):
    """Create metadata JSON for the trajectory data"""
    
    metadata = {
        "file_id": "franka_test_trajectory_v1",
        "description": "Test trajectories for Franka 7-DOF robot arm",
        "robot": "Franka Emika Panda",
        "columns": [
            {"name": "timestamp", "dtype": "float64", "unit": "seconds"},
            {"name": "joint_0", "dtype": "float32", "unit": "radians", "description": "Shoulder yaw"},
            {"name": "joint_1", "dtype": "float32", "unit": "radians", "description": "Shoulder pitch"},
            {"name": "joint_2", "dtype": "float32", "unit": "radians", "description": "Shoulder roll"},
            {"name": "joint_3", "dtype": "float32", "unit": "radians", "description": "Elbow"},
            {"name": "joint_4", "dtype": "float32", "unit": "radians", "description": "Wrist yaw"},
            {"name": "joint_5", "dtype": "float32", "unit": "radians", "description": "Wrist pitch"},
            {"name": "joint_6", "dtype": "float32", "unit": "radians", "description": "Wrist roll"},
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