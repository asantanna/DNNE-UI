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
  
  # Custom configuration
  python performance_profiler.py --iterations 20 --num-envs 1024 --mode detailed
        """
    )
    
    parser.add_argument('--mode', choices=['simple', 'detailed'], default='simple',
                        help='Profiling mode: simple (basic metrics) or detailed (function-level breakdown)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of training iterations (default: 10)')
    parser.add_argument('--num-envs', type=int, default=512,
                        help='Number of parallel environments (default: 512)')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout in seconds for each system (default: 300)')
    parser.add_argument('--systems', nargs='+', choices=['isaacgym', 'dnne', 'both'], 
                        default=['both'],
                        help='Which systems to profile (default: both)')
    
    args = parser.parse_args()
    
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
    print(f"Configuration: {args.iterations} iterations, {args.num_envs} environments")
    print(f"Timeout: {args.timeout}s per system")
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
        num_iterations=args.iterations,
        num_envs=args.num_envs,
        timeout=args.timeout
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
                    'iterations': args.iterations,
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