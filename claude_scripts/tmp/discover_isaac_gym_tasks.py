#!/usr/bin/env python3
"""
Discover available IsaacGymEnvs tasks for DNNE UI dropdown.
"""

import os
import json
from pathlib import Path

def discover_isaac_gym_tasks():
    """Discover all available IsaacGymEnvs tasks."""
    
    # Path to IsaacGymEnvs task configs
    task_dir = Path("/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs/isaacgymenvs/cfg/task")
    
    if not task_dir.exists():
        print(f"Error: Task directory not found at {task_dir}")
        return []
    
    # Get all YAML files (excluding directories)
    task_files = [f for f in task_dir.iterdir() if f.is_file() and f.suffix == '.yaml']
    
    # Extract task names (filename without extension)
    tasks = []
    for task_file in sorted(task_files):
        task_name = task_file.stem
        
        # Check if corresponding PPO config exists
        ppo_config = task_dir.parent / "train" / f"{task_name}PPO.yaml"
        has_ppo_config = ppo_config.exists()
        
        tasks.append({
            "name": task_name,
            "file": str(task_file),
            "has_ppo_config": has_ppo_config,
            "ppo_config_file": str(ppo_config) if has_ppo_config else None
        })
    
    return tasks

def main():
    """Main function to discover and display tasks."""
    tasks = discover_isaac_gym_tasks()
    
    print(f"Found {len(tasks)} IsaacGymEnvs tasks:\n")
    
    # Group by PPO config availability
    with_ppo = [t for t in tasks if t['has_ppo_config']]
    without_ppo = [t for t in tasks if not t['has_ppo_config']]
    
    print(f"Tasks with PPO configs ({len(with_ppo)}):")
    for task in with_ppo:
        print(f"  - {task['name']}")
    
    if without_ppo:
        print(f"\nTasks without PPO configs ({len(without_ppo)}):")
        for task in without_ppo:
            print(f"  - {task['name']}")
    
    # Save as JSON for DNNE UI to use
    output_file = Path(__file__).parent / "isaac_gym_tasks.json"
    with open(output_file, 'w') as f:
        json.dump({
            "tasks": tasks,
            "task_names": [t['name'] for t in tasks],
            "ppo_compatible_tasks": [t['name'] for t in with_ppo]
        }, f, indent=2)
    
    print(f"\nTask list saved to: {output_file}")

if __name__ == "__main__":
    main()