#!/usr/bin/env python3
"""
Generate test CSV data for Franka robot arm in joint torque mode
Creates two types of exploration data:
1. Smooth sinusoidal trajectories
2. Step-and-hold patterns for steady-state exploration

For use with simplified 3-joint control: joints 0 (base), 1 (shoulder), 3 (elbow)
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path

# Joint configuration for simplified control
CONTROLLED_JOINTS = [0, 1, 3]  # Base, shoulder, elbow
FROZEN_JOINTS = [2, 4, 5, 6]   # Shoulder roll, forearm roll, wrist pitch/roll
TOTAL_ARM_JOINTS = 7

# Safe torque limits for training (Nm)
TORQUE_LIMITS = {
    0: 1.0,  # Base
    1: 1.0,  # Shoulder
    3: 1.0,  # Elbow
}

def generate_sinusoidal_torques(num_samples=10000, frequency_hz=100.0, 
                                base_freq=0.1, shoulder_freq=0.15, elbow_freq=0.08,
                                add_timestamp=False):
    """
    Generate smooth sinusoidal torque commands for exploration.
    
    Args:
        num_samples: Number of data points to generate
        frequency_hz: Data collection frequency (Hz)
        base_freq: Frequency for base joint oscillation (Hz)
        shoulder_freq: Frequency for shoulder joint oscillation (Hz)
        elbow_freq: Frequency for elbow joint oscillation (Hz)
        add_timestamp: Whether to include timestamp column
    
    Returns:
        DataFrame with torque commands
    """
    # Time array
    duration = num_samples / frequency_hz
    t = np.linspace(0, duration, num_samples)
    
    # Random phase offsets for variety
    phase_base = np.random.uniform(0, 2*np.pi)
    phase_shoulder = np.random.uniform(0, 2*np.pi)
    phase_elbow = np.random.uniform(0, 2*np.pi)
    
    # Generate sinusoidal torques with different frequencies
    torque_base = TORQUE_LIMITS[0] * 0.7 * np.sin(2*np.pi*base_freq*t + phase_base)
    torque_shoulder = TORQUE_LIMITS[1] * 0.7 * np.sin(2*np.pi*shoulder_freq*t + phase_shoulder)
    torque_elbow = TORQUE_LIMITS[3] * 0.7 * np.sin(2*np.pi*elbow_freq*t + phase_elbow)
    
    # Add small random noise for exploration
    noise_scale = 0.05
    torque_base += np.random.normal(0, noise_scale, num_samples)
    torque_shoulder += np.random.normal(0, noise_scale, num_samples)
    torque_elbow += np.random.normal(0, noise_scale, num_samples)
    
    # Clip to limits
    torque_base = np.clip(torque_base, -TORQUE_LIMITS[0], TORQUE_LIMITS[0])
    torque_shoulder = np.clip(torque_shoulder, -TORQUE_LIMITS[1], TORQUE_LIMITS[1])
    torque_elbow = np.clip(torque_elbow, -TORQUE_LIMITS[3], TORQUE_LIMITS[3])
    
    # Create DataFrame with all joint torques (7 total)
    data = {
        'torque_joint0': torque_base,
        'torque_joint1': torque_shoulder,
        'torque_joint2': np.zeros(num_samples),  # Frozen
        'torque_joint3': torque_elbow,
        'torque_joint4': np.zeros(num_samples),  # Frozen
        'torque_joint5': np.zeros(num_samples),  # Frozen
        'torque_joint6': np.zeros(num_samples),  # Frozen
    }
    
    # Optionally add timestamp as first column
    if add_timestamp:
        data = {'timestamp': t, **data}
    
    return pd.DataFrame(data)

def generate_step_hold_torques(num_samples=10000, frequency_hz=100.0, 
                               min_hold_time=1.0, max_hold_time=2.0,
                               add_timestamp=False):
    """
    Generate step-and-hold torque commands for steady-state exploration.
    
    Args:
        num_samples: Number of data points to generate
        frequency_hz: Data collection frequency (Hz)
        min_hold_time: Minimum time to hold each torque (seconds)
        max_hold_time: Maximum time to hold each torque (seconds)
        add_timestamp: Whether to include timestamp column
    
    Returns:
        DataFrame with torque commands
    """
    # Time array
    duration = num_samples / frequency_hz
    t = np.linspace(0, duration, num_samples)
    
    # Initialize torque arrays
    torque_base = np.zeros(num_samples)
    torque_shoulder = np.zeros(num_samples)
    torque_elbow = np.zeros(num_samples)
    
    # Generate step changes at random intervals
    current_time = 0.0
    sample_idx = 0
    
    while sample_idx < num_samples:
        # Random hold duration
        hold_duration = np.random.uniform(min_hold_time, max_hold_time)
        hold_samples = int(hold_duration * frequency_hz)
        
        # Random torque values for this period
        base_torque = np.random.uniform(-TORQUE_LIMITS[0], TORQUE_LIMITS[0])
        shoulder_torque = np.random.uniform(-TORQUE_LIMITS[1], TORQUE_LIMITS[1])
        elbow_torque = np.random.uniform(-TORQUE_LIMITS[3], TORQUE_LIMITS[3])
        
        # Apply to the arrays
        end_idx = min(sample_idx + hold_samples, num_samples)
        torque_base[sample_idx:end_idx] = base_torque
        torque_shoulder[sample_idx:end_idx] = shoulder_torque
        torque_elbow[sample_idx:end_idx] = elbow_torque
        
        sample_idx = end_idx
    
    # Create DataFrame with all joint torques
    data = {
        'torque_joint0': torque_base,
        'torque_joint1': torque_shoulder,
        'torque_joint2': np.zeros(num_samples),  # Frozen
        'torque_joint3': torque_elbow,
        'torque_joint4': np.zeros(num_samples),  # Frozen
        'torque_joint5': np.zeros(num_samples),  # Frozen
        'torque_joint6': np.zeros(num_samples),  # Frozen
    }
    
    # Optionally add timestamp as first column
    if add_timestamp:
        data = {'timestamp': t, **data}
    
    return pd.DataFrame(data)

def add_metadata(df, mode, params, add_timestamp=False):
    """Add metadata as JSON comment in first row"""
    columns_desc = {
        "torque_joint0": "Base joint torque (Nm)",
        "torque_joint1": "Shoulder joint torque (Nm)",
        "torque_joint2": "Shoulder roll torque (Nm) - frozen",
        "torque_joint3": "Elbow joint torque (Nm)",
        "torque_joint4": "Forearm roll torque (Nm) - frozen",
        "torque_joint5": "Wrist pitch torque (Nm) - frozen",
        "torque_joint6": "Wrist roll torque (Nm) - frozen"
    }
    
    # Add timestamp description if included
    if add_timestamp:
        columns_desc = {"timestamp": "Time in seconds", **columns_desc}
    
    metadata = {
        "description": f"Franka 3-joint torque control exploration data ({mode})",
        "controlled_joints": CONTROLLED_JOINTS,
        "frozen_joints": FROZEN_JOINTS,
        "torque_limits_nm": TORQUE_LIMITS,
        "frequency_hz": params.get("frequency_hz", 100.0),
        "mode": mode,
        "mode_params": params,
        "columns": columns_desc,
        "has_timestamp": add_timestamp
    }
    
    # Add metadata as comment in CSV (will be ignored by pd.read_csv)
    return metadata, df

def main():
    parser = argparse.ArgumentParser(description='Generate Franka torque control exploration data')
    parser.add_argument('--mode', choices=['sinusoidal', 'step_hold', 'both'], 
                       default='both', help='Generation mode')
    parser.add_argument('--samples', type=int, default=10000, 
                       help='Number of samples to generate')
    parser.add_argument('--frequency', type=float, default=100.0, 
                       help='Data collection frequency in Hz')
    parser.add_argument('--output_dir', default='test_data', 
                       help='Output directory for CSV files')
    
    # Sinusoidal mode parameters
    parser.add_argument('--base_freq', type=float, default=0.1,
                       help='Base joint oscillation frequency (Hz)')
    parser.add_argument('--shoulder_freq', type=float, default=0.15,
                       help='Shoulder joint oscillation frequency (Hz)')
    parser.add_argument('--elbow_freq', type=float, default=0.08,
                       help='Elbow joint oscillation frequency (Hz)')
    
    # Step-hold mode parameters
    parser.add_argument('--min_hold', type=float, default=1.0,
                       help='Minimum hold time in seconds')
    parser.add_argument('--max_hold', type=float, default=2.0,
                       help='Maximum hold time in seconds')
    
    # Add timestamp option
    parser.add_argument('--add-timestamp', action='store_true',
                       help='Include timestamp column in output CSV')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate sinusoidal data
    if args.mode in ['sinusoidal', 'both']:
        print(f"Generating sinusoidal exploration data ({args.samples} samples)...")
        df_sin = generate_sinusoidal_torques(
            num_samples=args.samples,
            frequency_hz=args.frequency,
            base_freq=args.base_freq,
            shoulder_freq=args.shoulder_freq,
            elbow_freq=args.elbow_freq,
            add_timestamp=args.add_timestamp
        )
        
        # Save with metadata
        params = {
            "frequency_hz": args.frequency,
            "base_freq": args.base_freq,
            "shoulder_freq": args.shoulder_freq,
            "elbow_freq": args.elbow_freq
        }
        metadata, df_sin = add_metadata(df_sin, "sinusoidal", params, args.add_timestamp)
        
        # Save CSV
        output_file = output_dir / "franka_sinusoidal_exploration_001.csv"
        df_sin.to_csv(output_file, index=False)
        
        # Save metadata as separate JSON
        import json
        metadata_file = output_dir / "franka_sinusoidal_exploration_001_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved: {output_file}")
        print(f"  Metadata: {metadata_file}")
    
    # Generate step-hold data
    if args.mode in ['step_hold', 'both']:
        print(f"Generating step-hold exploration data ({args.samples} samples)...")
        df_step = generate_step_hold_torques(
            num_samples=args.samples,
            frequency_hz=args.frequency,
            min_hold_time=args.min_hold,
            max_hold_time=args.max_hold,
            add_timestamp=args.add_timestamp
        )
        
        # Save with metadata
        params = {
            "frequency_hz": args.frequency,
            "min_hold_time": args.min_hold,
            "max_hold_time": args.max_hold
        }
        metadata, df_step = add_metadata(df_step, "step_hold", params, args.add_timestamp)
        
        # Save CSV
        output_file = output_dir / "franka_step_hold_exploration_001.csv"
        df_step.to_csv(output_file, index=False)
        
        # Save metadata as separate JSON
        import json
        metadata_file = output_dir / "franka_step_hold_exploration_001_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved: {output_file}")
        print(f"  Metadata: {metadata_file}")
    
    print("\nGeneration complete!")
    print(f"Files saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()