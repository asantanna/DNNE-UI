#!/usr/bin/env python3
"""
Performance Profiler - Main Entry Point

Orchestrates performance profiling of IsaacGymEnvs and DNNE systems.
Supports both simple and detailed profiling modes.
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add profiling directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main entry point for performance profiling"""
    parser = argparse.ArgumentParser(
        description='Performance Profiler for IsaacGymEnvs and DNNE comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple profiling (basic metrics only)
  python performance_profiler.py --mode simple
  
  # Detailed profiling (function-level breakdown)
  python performance_profiler.py --mode detailed
  
  # Visual mode profiling (shows rendering, slower)
  python performance_profiler.py --mode simple --visual
  
  # Custom configuration
  python performance_profiler.py --epochs 20 --num-envs 1024 --mode detailed
        """
    )
    
    parser.add_argument('--mode', choices=['simple', 'detailed'], default='simple',
                        help='Profiling mode: simple (basic metrics) or detailed (function-level breakdown)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs to run (default: use workflow value)')
    parser.add_argument('--num-envs', type=int, default=512,
                        help='Number of parallel environments (default: 512)')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout in seconds for each system (default: 300)')
    parser.add_argument('--systems', nargs='+', choices=['isaacgym', 'dnne', 'both'], 
                        default=['both'],
                        help='Which systems to profile (default: both)')
    parser.add_argument('--visual', action='store_true',
                        help='Run in visual mode with rendering enabled (slower but shows environments)')
    
    # PPO cycle debugging options
    parser.add_argument('--ppo-cycle-debug', action='store_true',
                        help='Enable PPO cycle debugging mode (captures detailed computation values)')
    parser.add_argument('--stop-after-cycle', type=int, default=None,
                        help='Stop after N PPO cycles (16 steps each) - requires --ppo-cycle-debug')
    parser.add_argument('--fixed-seed', type=int, default=None,
                        help='Use fixed seed for deterministic comparison (e.g., --fixed-seed 42)')
    parser.add_argument('--capture-values', action='store_true',
                        help='Capture and save computation values for comparison - requires --ppo-cycle-debug')
    
    args = parser.parse_args()
    
    # Validate PPO cycle debug options
    if args.stop_after_cycle and not args.ppo_cycle_debug:
        parser.error("--stop-after-cycle requires --ppo-cycle-debug")
    if args.capture_values and not args.ppo_cycle_debug:
        parser.error("--capture-values requires --ppo-cycle-debug")
    
    # Check environment
    if os.environ.get("CONDA_DEFAULT_ENV") != "DNNE_PY38":
        print("⚠️  Warning: DNNE_PY38 conda environment not active")
        print("   Activate with: source /home/asantanna/miniconda/bin/activate DNNE_PY38")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print("🚀 PERFORMANCE PROFILER")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Configuration: {args.num_envs} environments")
    if args.epochs:
        print(f"Epochs override: {args.epochs}")
    print(f"Timeout: {args.timeout}s per system")
    if args.visual:
        print(f"Visual mode: ENABLED (rendering will show)")
    if args.ppo_cycle_debug:
        print(f"PPO Cycle Debug: ENABLED")
        if args.stop_after_cycle:
            print(f"  - Stop after {args.stop_after_cycle} cycle(s): YES")
        if args.fixed_seed is not None:
            print(f"  - Fixed seed: {args.fixed_seed}")
        if args.capture_values:
            print(f"  - Capture values: YES")
    print()
    
    # Import helper modules
    try:
        from _profile_runner import ProfileRunner
        from _profile_analyzer import ProfileAnalyzer
        from _profile_formatter import ProfileFormatter
    except ImportError as e:
        print(f"❌ Failed to import helper modules: {e}")
        print("   Make sure all helper files are in the same directory")
        sys.exit(1)
    
    # Determine which systems to profile
    systems_to_profile = []
    if 'both' in args.systems:
        systems_to_profile = ['isaacgym', 'dnne']
    else:
        systems_to_profile = args.systems
    
    # Run profiling
    runner = ProfileRunner(
        num_envs=args.num_envs,
        timeout=args.timeout,
        override_epochs=args.epochs,
        visual=args.visual,
        ppo_cycle_debug=args.ppo_cycle_debug,
        stop_after_cycle=args.stop_after_cycle,
        fixed_seed=args.fixed_seed,
        capture_values=args.capture_values
    )
    
    results = {}
    
    # Profile each system
    for system in systems_to_profile:
        print(f"\n{'='*60}")
        print(f"Profiling {system.upper()}...")
        print(f"{'='*60}")
        
        if system == 'isaacgym':
            profile_data = runner.profile_isaacgymenvs()
        else:
            profile_data = runner.profile_dnne()
        
        if profile_data:
            # Analyze profile if in detailed mode
            if args.mode == 'detailed':
                analyzer = ProfileAnalyzer()
                detailed_metrics = analyzer.analyze_profile(
                    system=system,
                    prof_file=profile_data['prof_file'],
                    basic_metrics=profile_data
                )
                results[system] = detailed_metrics
            else:
                results[system] = profile_data
        else:
            print(f"❌ Failed to profile {system}")
    
    # Format and display results
    if results:
        print("\n" + "="*70)
        formatter = ProfileFormatter(mode=args.mode)
        formatter.format_comparison(results)
        
        # Save combined results
        output_file = Path('/tmp/performance_comparison_results.json')
        with open(output_file, 'w') as f:
            json.dump({
                'config': {
                    'mode': args.mode,
                    'epochs': args.epochs,
                    'num_envs': args.num_envs,
                    'timeout': args.timeout
                },
                'results': results
            }, f, indent=2)
        
        print(f"\n💾 Full results saved to {output_file}")
    else:
        print("\n❌ No results to display")
        sys.exit(1)
    
    print("\n✅ Profiling complete!")

if __name__ == "__main__":
    main()