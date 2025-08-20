"""
DNNE Argument Parser Template
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  CRITICAL: SYNCHRONIZATION REQUIREMENT ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When adding/modifying/removing ANY command-line argument in this file:

1. YOU MUST update runner_args.json in the same directory
2. The JSON file defines the UI for these arguments
3. DNNE will FAIL the export if this file is newer than runner_args.json
4. This ensures the UI always matches available command-line options

To update runner_args.json after changes:
  - Add corresponding UI configuration for new arguments
  - Remove UI configuration for deleted arguments  
  - Update descriptions/types for modified arguments
  - Run tests to verify: python -m pytest tests/test_runner_args_sync.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse

def create_parser():
    """Create the argument parser for DNNE runner"""
    parser = argparse.ArgumentParser(description='DNNE Generated Training')
    
    # Logging arguments
    parser.add_argument('--verbose', '-v', nargs='?', const='all', default=None,
                       help='Enable verbose logging (INFO level). Optional: comma-separated subsystems or node IDs (e.g., "training,data,42" or "55")')
    parser.add_argument('--debug', '-d', nargs='?', const='all', default=None,
                       help='Enable debug logging (DEBUG level). Optional: comma-separated subsystems or node IDs (e.g., "rl,robotics,42" or "66")')
    
    # Checkpoint arguments
    parser.add_argument('--save-checkpoint', action='store_true',
                       help='Enable checkpoint saving')
    parser.add_argument('--out-dir', type=str, default='runs/singles',
                       help='Output directory for checkpoints and other outputs (default: runs/singles)')
    parser.add_argument('--load-checkpoint', type=str,
                       help='Directory to load checkpoints from (expects node_<id> subdirectories)')
    
    # Training control arguments
    parser.add_argument('--timeout', type=str,
                       help='Training duration (e.g., 5, 30s, 5m, 1h30m)')
    parser.add_argument('--visual', action='store_true',
                       help='Enable visual mode (overrides headless setting)')
    parser.add_argument('--headless', action='store_true',
                       help='Force headless mode (default)')
    parser.add_argument('--inference', action='store_true',
                       help='Run in inference mode (no training, no gradients)')
    parser.add_argument('--dnne-profiling', action='store_true',
                       help='Enable profiling for C++ operations (Isaac Gym)')
    
    # Node-specific overrides
    parser.add_argument('--epochs', type=str, default=None,
                       help='Override max epochs for EpochTracker nodes (e.g., --epochs 10 or --epochs 55:10,56:20)')
    parser.add_argument('--max-iterations', type=str, default=None,
                       help='Override max iterations for PPOAgent nodes (e.g., --max-iterations 1000 or --max-iterations 66:5000,67:10000)')
    parser.add_argument('--learning-rate', type=str, default=None,
                       help='Override learning rate for SGDOptimizer nodes (e.g., --learning-rate 0.01 or --learning-rate 68:0.001,69:0.01)')
    parser.add_argument('--batch-size', type=str, default=None,
                       help='Override batch size for BatchSampler nodes (e.g., --batch-size 32 or --batch-size 38:64,39:128)')
    
    # Reproducibility
    parser.add_argument('--fixed-seed', type=int, default=None,
                       help='Use fixed random seed for deterministic execution')
    
    # Advanced overrides
    parser.add_argument('--override', type=str, default=None,
                       help='Override node configuration values. Use node IDs or subsystems (e.g., --override 56:checkpoint_enabled=True,training:learning_rate=0.001,rl:gamma=0.99)')
    
    # Telemetry
    parser.add_argument('--enable-telemetry', type=str, nargs='?', const='all', default=None,
                       help='Enable telemetry reporting. Optional: comma-separated node IDs, subsystems, or "all" (e.g., --enable-telemetry training,10,11 or --enable-telemetry rl)')
    
    return parser

def process_args(args):
    """Post-process parsed arguments to convert types appropriately"""
    # Convert node-specific arguments that should be integers when no ':' present
    for arg_name in ['epochs', 'max_iterations', 'batch_size']:
        value = getattr(args, arg_name.replace('-', '_'), None)
        if value and ':' not in value:
            try:
                setattr(args, arg_name.replace('-', '_'), int(value))
            except ValueError:
                pass  # Keep as string if conversion fails
    
    # Convert learning_rate if no ':' present
    if args.learning_rate and ':' not in args.learning_rate:
        try:
            args.learning_rate = float(args.learning_rate)
        except ValueError:
            pass  # Keep as string if conversion fails
    
    return args