#!/usr/bin/env python3
"""
Profile Analyzer - Extracts detailed metrics from cProfile output

Analyzes .prof files to extract function-level timing information
for detailed performance breakdown.
"""

import pstats
import json
from pathlib import Path
from collections import defaultdict

class ProfileAnalyzer:
    """Analyzes cProfile output for detailed metrics"""
    
    # Function patterns to search for in profiles
    ISAACGYM_PATTERNS = {
        'env_step_total': ['step', 'env_step'],
        'gym_simulate': ['simulate', 'gym_simulate'],
        'compute_obs': ['compute_obs', 'get_observations'],
        'compute_reward': ['compute_reward', 'calculate_metrics'],
        'reset_envs': ['reset', 'reset_envs', 'reset_idx'],
        'collect_rollout': ['collect_rollout', 'rollout'],
        'compute_returns': ['compute_returns', 'calculate_returns'],
        'policy_forward': ['forward', 'policy_forward', 'actor_critic'],
        'policy_backward': ['backward', 'loss.backward'],
        'optimizer_step': ['optimizer.step', 'adam.step']
    }
    
    DNNE_PATTERNS = {
        'env_step_total': ['step', 'env_step'],
        'gym_simulate': ['simulate', 'gym_simulate'],
        'compute_obs': ['compute_obs', 'get_observations'],
        'compute_reward': ['compute_reward', 'calculate_metrics'],
        'reset_envs': ['reset', 'reset_envs'],
        'collect_rollout': ['collect_rollout', 'rollout'],
        'compute_returns': ['compute_returns', 'calculate_returns'],
        'policy_forward': ['forward', 'network_forward'],
        'policy_backward': ['backward', 'compute_gradients'],
        'optimizer_step': ['optimizer_step', 'sgd_step'],
        'queue_operations': ['queue', 'get_queue', 'put_queue', 'QueueNode']
    }
    
    def analyze_profile(self, system, prof_file, basic_metrics):
        """
        Analyze a profile file to extract detailed metrics
        
        Args:
            system: 'isaacgym' or 'dnne'
            prof_file: Path to .prof file
            basic_metrics: Basic metrics dict from profile runner
            
        Returns:
            Enhanced metrics dict with detailed timings
        """
        if not Path(prof_file).exists():
            print(f"  ⚠️  Profile file not found: {prof_file}")
            return basic_metrics
        
        print(f"  📊 Analyzing {system} profile...")
        
        # Load profile stats
        stats = pstats.Stats(prof_file)
        
        # Get patterns for this system
        patterns = self.ISAACGYM_PATTERNS if system == 'isaacgym' else self.DNNE_PATTERNS
        
        # Extract timings from cProfile
        timings = {}
        function_times = self._extract_function_times(stats)
        
        for metric_name, search_patterns in patterns.items():
            time_ms = self._find_matching_time(function_times, search_patterns)
            timings[metric_name] = time_ms
        
        # Load and merge C++ timing data if available
        cpp_timings = self._load_cpp_timings(system)
        if cpp_timings:
            print(f"  📊 Found C++ timing data with {len(cpp_timings)} metrics")
            # Map C++ timings to our metric names
            cpp_mapping = {
                'gym.simulate': 'gym_simulate',
                'gym.fetch_results': 'gym_fetch_results',
                'gym.refresh_dof_state_tensor': 'refresh_tensors',
                'gym.refresh_actor_root_state_tensor': 'refresh_tensors',
                'gym.step_graphics': 'step_graphics',
                'gym.draw_viewer': 'draw_viewer'
            }
            
            for cpp_name, cpp_data in cpp_timings.items():
                metric_name = cpp_mapping.get(cpp_name, cpp_name)
                if metric_name in timings and timings[metric_name] is None:
                    # Use C++ timing if Python timing not found
                    # Use average time per call, not total
                    timings[metric_name] = cpp_data.get('avg_ms', cpp_data.get('average_ms', cpp_data['total_ms'] / cpp_data.get('count', 1)))
                elif metric_name == 'refresh_tensors':
                    # Aggregate refresh tensor calls
                    if timings.get(metric_name) is None:
                        timings[metric_name] = 0
                    # Use average time per call
                    avg_ms = cpp_data.get('avg_ms', cpp_data.get('average_ms', cpp_data['total_ms'] / cpp_data.get('count', 1)))
                    timings[metric_name] += avg_ms
        
        # Create enhanced metrics
        enhanced_metrics = basic_metrics.copy()
        enhanced_metrics['timings'] = timings
        
        # Calculate additional summary metrics
        enhanced_metrics['summary'] = {
            'total_time': basic_metrics['total_time'],
            'init_time': self._estimate_init_time(timings, basic_metrics),
            'step_count': basic_metrics['step_count'],
            'steps_per_sec': basic_metrics['steps_per_sec'],
            'iterations_per_sec': basic_metrics['num_iterations'] / basic_metrics['total_time']
        }
        
        # Add timing percentages
        total_tracked_time = sum(t for t in timings.values() if t is not None)
        enhanced_metrics['timing_percentages'] = {}
        for name, time_ms in timings.items():
            if time_ms is not None and total_tracked_time > 0:
                percentage = (time_ms / 1000) / basic_metrics['total_time'] * 100
                enhanced_metrics['timing_percentages'][name] = percentage
        
        print(f"  ✅ Found {len([t for t in timings.values() if t is not None])} timing metrics")
        
        # Save detailed report
        self.save_detailed_report(system, enhanced_metrics)
        
        return enhanced_metrics
    
    def _extract_function_times(self, stats):
        """Extract all function times from pstats"""
        function_times = {}
        
        # stats.stats is a dict of ((filename, line, function), cumtime, ...)
        for func_key, func_stats in stats.stats.items():
            filename, line_num, func_name = func_key
            cumulative_time = func_stats[3]  # cumulative time in seconds
            
            # Store by function name and full key
            function_times[func_name] = cumulative_time * 1000  # Convert to ms
            
            # Also store with module prefix if available
            if '/' in filename:
                module = filename.split('/')[-1].replace('.py', '')
                full_name = f"{module}.{func_name}"
                function_times[full_name] = cumulative_time * 1000
        
        return function_times
    
    def _find_matching_time(self, function_times, patterns):
        """Find the best matching function time for given patterns"""
        for pattern in patterns:
            # Direct match
            if pattern in function_times:
                return function_times[pattern]
            
            # Partial match
            for func_name, time_ms in function_times.items():
                if pattern.lower() in func_name.lower():
                    return time_ms
        
        return None  # Not found
    
    def _estimate_init_time(self, timings, basic_metrics):
        """Estimate initialization time"""
        # For now, return a placeholder
        # In a real implementation, we'd look for __init__ or setup functions
        return 0.0
    
    def _load_cpp_timings(self, system):
        """Load C++ timing data from wrapper output"""
        if system == 'isaacgym':
            timing_file = Path('/tmp/isaacgym_cpp_timings.json')
        else:
            timing_file = Path('/tmp/dnne_cpp_timings.json')
        
        if timing_file.exists():
            try:
                with open(timing_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"  ⚠️  Failed to load C++ timings: {e}")
        
        return None
    
    def save_detailed_report(self, system, enhanced_metrics, output_dir='/tmp'):
        """Save a detailed analysis report"""
        report_file = Path(output_dir) / f"{system}_detailed_analysis.json"
        
        with open(report_file, 'w') as f:
            json.dump(enhanced_metrics, f, indent=2)
        
        print(f"  💾 Detailed report saved to {report_file}")
        
        # Also create a human-readable summary
        summary_file = Path(output_dir) / f"{system}_analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Detailed Performance Analysis: {system.upper()}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Summary Metrics:\n")
            f.write(f"  Total Time: {enhanced_metrics['total_time']:.2f}s\n")
            f.write(f"  Steps/sec: {enhanced_metrics['steps_per_sec']:.1f}\n")
            f.write(f"  Step Count: {enhanced_metrics['step_count']}\n\n")
            
            f.write("Timing Breakdown (ms):\n")
            if 'timings' in enhanced_metrics:
                for name, time_ms in enhanced_metrics['timings'].items():
                    if time_ms is not None:
                        f.write(f"  {name:25}: {time_ms:8.2f}ms")
                        if 'timing_percentages' in enhanced_metrics:
                            pct = enhanced_metrics['timing_percentages'].get(name, 0)
                            f.write(f" ({pct:5.1f}%)")
                        f.write("\n")
                    else:
                        f.write(f"  {name:25}: Not found\n")
        
        print(f"  📄 Summary saved to {summary_file}")