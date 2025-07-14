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
        'optimizer_step': ['optimizer.step', 'adam.step'],
        'draw_viewer': ['draw_viewer', 'render_viewer', 'render'],
        'step_graphics': ['step_graphics', 'graphics_step'],
        'update_viewer': ['update_viewer', 'viewer_update']
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
        'queue_operations': ['queue', 'get_queue', 'put_queue', 'QueueNode'],
        'draw_viewer': ['draw_viewer', 'render_viewer', 'render'],
        'step_graphics': ['step_graphics', 'graphics_step'],
        'update_viewer': ['update_viewer', 'viewer_update']
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
        function_data = self._extract_function_data(stats)
        
        for metric_name, search_patterns in patterns.items():
            time_ms, call_count = self._find_matching_data(function_data, search_patterns)
            timings[metric_name] = time_ms
            # Store call counts for render metrics
            if metric_name in ['draw_viewer', 'step_graphics', 'update_viewer']:
                timings[f'{metric_name}_count'] = call_count
        
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
        
        # Update steps_per_sec with accurate first/last timing if available
        if cpp_timings and 'step_rate_info' in cpp_timings:
            step_rate_info = cpp_timings['step_rate_info']
            if 'steps_per_second' in step_rate_info:
                print(f"  📊 Using accurate step rate from first/last timing: {step_rate_info['steps_per_second']:.1f} steps/sec")
                enhanced_metrics['steps_per_sec'] = step_rate_info['steps_per_second']
                enhanced_metrics['step_rate_info'] = step_rate_info
        
        # Calculate additional summary metrics
        enhanced_metrics['summary'] = {
            'total_time': basic_metrics['total_time'],
            'init_time': self._estimate_init_time(timings, basic_metrics),
            'step_count': basic_metrics['step_count'],
            'steps_per_sec': enhanced_metrics.get('steps_per_sec', basic_metrics['steps_per_sec']),
            'epochs_per_sec': basic_metrics['num_epochs'] / basic_metrics['total_time']
        }
        
        # Add timing percentages
        total_tracked_time = sum(t for t in timings.values() if t is not None)
        enhanced_metrics['timing_percentages'] = {}
        for name, time_ms in timings.items():
            if time_ms is not None and total_tracked_time > 0:
                percentage = (time_ms / 1000) / basic_metrics['total_time'] * 100
                enhanced_metrics['timing_percentages'][name] = percentage
        
        print(f"  ✅ Found {len([t for t in timings.values() if t is not None])} timing metrics")
        
        # Extract render count metrics from cProfile data
        render_metrics = self._extract_render_metrics(timings, basic_metrics)
        enhanced_metrics['render_metrics'] = render_metrics
        
        # Save detailed report
        self.save_detailed_report(system, enhanced_metrics)
        
        return enhanced_metrics
    
    def _extract_function_data(self, stats):
        """Extract all function times and call counts from pstats"""
        function_data = {}
        
        # stats.stats is a dict of ((filename, line, function), (callcount, reccallcount, totaltime, cumtime))
        for func_key, func_stats in stats.stats.items():
            filename, line_num, func_name = func_key
            call_count = func_stats[0]  # number of calls
            cumulative_time = func_stats[3]  # cumulative time in seconds
            
            # Store by function name and full key
            function_data[func_name] = {
                'time_ms': cumulative_time * 1000,  # Convert to ms
                'call_count': call_count
            }
            
            # Also store with module prefix if available
            if '/' in filename:
                module = filename.split('/')[-1].replace('.py', '')
                full_name = f"{module}.{func_name}"
                function_data[full_name] = {
                    'time_ms': cumulative_time * 1000,
                    'call_count': call_count
                }
        
        return function_data
    
    def _find_matching_data(self, function_data, patterns):
        """Find the best matching function data for given patterns"""
        for pattern in patterns:
            # Direct match
            if pattern in function_data:
                data = function_data[pattern]
                return data['time_ms'], data['call_count']
            
            # Partial match
            for func_name, data in function_data.items():
                if pattern.lower() in func_name.lower():
                    return data['time_ms'], data['call_count']
        
        return None, 0  # Not found
    
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
    
    def _extract_render_metrics(self, timings, basic_metrics):
        """Extract render-related metrics from cProfile timing data"""
        render_metrics = {
            'total_renders': 0,
            'renders_per_sec': 0.0,
            'total_step_graphics': 0,
            'step_graphics_per_sec': 0.0,
            'avg_render_time_ms': 0.0,
            'avg_step_graphics_time_ms': 0.0
        }
        
        if not timings:
            return render_metrics
            
        # Extract draw_viewer metrics from cProfile data
        if 'draw_viewer_count' in timings and timings['draw_viewer_count'] > 0:
            render_metrics['total_renders'] = timings['draw_viewer_count']
            if basic_metrics['total_time'] > 0:
                render_metrics['renders_per_sec'] = render_metrics['total_renders'] / basic_metrics['total_time']
            # Calculate average time per render if timing data available
            if timings.get('draw_viewer') is not None and render_metrics['total_renders'] > 0:
                render_metrics['avg_render_time_ms'] = timings['draw_viewer'] / render_metrics['total_renders']
                
        # Extract step_graphics metrics from cProfile data
        if 'step_graphics_count' in timings and timings['step_graphics_count'] > 0:
            render_metrics['total_step_graphics'] = timings['step_graphics_count']
            if basic_metrics['total_time'] > 0:
                render_metrics['step_graphics_per_sec'] = render_metrics['total_step_graphics'] / basic_metrics['total_time']
            # Calculate average time per step_graphics if timing data available
            if timings.get('step_graphics') is not None and render_metrics['total_step_graphics'] > 0:
                render_metrics['avg_step_graphics_time_ms'] = timings['step_graphics'] / render_metrics['total_step_graphics']
        
        # Also check update_viewer as a fallback render metric
        if render_metrics['total_renders'] == 0 and 'update_viewer_count' in timings:
            render_metrics['total_renders'] = timings['update_viewer_count']
            if basic_metrics['total_time'] > 0:
                render_metrics['renders_per_sec'] = render_metrics['total_renders'] / basic_metrics['total_time']
            if timings.get('update_viewer') is not None and render_metrics['total_renders'] > 0:
                render_metrics['avg_render_time_ms'] = timings['update_viewer'] / render_metrics['total_renders']
        
        return render_metrics
    
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
            f.write(f"  Step Count: {enhanced_metrics['step_count']}\n")
            
            # Add step rate calculation details if available
            if 'step_rate_info' in enhanced_metrics:
                info = enhanced_metrics['step_rate_info']
                f.write(f"\nStep Rate Calculation:\n")
                f.write(f"  Method: First/Last step timing\n")
                f.write(f"  Elapsed Time: {info['elapsed_time']:.3f}s (excluding init/cleanup)\n")
                f.write(f"  Total Env Steps: {info['total_env_steps']}\n")
                f.write(f"  Accurate Rate: {info['steps_per_second']:.1f} steps/sec\n")
            
            # Add render metrics if available
            if 'render_metrics' in enhanced_metrics:
                render = enhanced_metrics['render_metrics']
                f.write(f"\nRender Metrics:\n")
                f.write(f"  Total Renders: {render['total_renders']}\n")
                f.write(f"  Renders/sec: {render['renders_per_sec']:.1f}\n")
                f.write(f"  Total Step Graphics: {render['total_step_graphics']}\n")
                f.write(f"  Step Graphics/sec: {render['step_graphics_per_sec']:.1f}\n")
                if render['avg_render_time_ms'] > 0:
                    f.write(f"  Avg Render Time: {render['avg_render_time_ms']:.2f}ms\n")
            
            f.write("\n")
            
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