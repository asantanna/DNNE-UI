#!/usr/bin/env python3
"""
Profile Formatter - Formats profiling results into readable tables

Generates the detailed comparison table in the original format.
"""

import json
from pathlib import Path

class ProfileFormatter:
    """Formats profiling results for display"""
    
    def __init__(self, mode='simple'):
        self.mode = mode
    
    def format_comparison(self, results):
        """
        Format comparison results into a readable table
        
        Args:
            results: Dict with 'isaacgym' and/or 'dnne' results
        """
        if self.mode == 'simple':
            self._format_simple_comparison(results)
        else:
            self._format_detailed_comparison(results)
    
    def _format_simple_comparison(self, results):
        """Format basic performance comparison"""
        print("\n📊 PERFORMANCE COMPARISON")
        print("=" * 60)
        
        # Check what results we have
        igenv = results.get('isaacgym', {})
        dnne = results.get('dnne', {})
        
        if not igenv and not dnne:
            print("❌ No results to display")
            return
        
        # Header
        if igenv and dnne:
            print(f"{'Metric':30} {'IsaacGymEnvs':>15} {'DNNE':>15}")
        elif igenv:
            print(f"{'Metric':30} {'IsaacGymEnvs':>15}")
        else:
            print(f"{'Metric':30} {'DNNE':>15}")
        
        print("-" * 60)
        
        # Main metrics
        metrics = [
            ('Total Time (s)', 'total_time', '.2f'),
            ('Steps/sec', 'steps_per_sec', '.1f'),
            ('Total Steps', 'step_count', 'd'),
            ('Epochs', 'num_epochs', 'd')
        ]
        
        for label, key, fmt in metrics:
            row = f"{label:<30}"
            if igenv:
                value = igenv.get(key, 0)
                row += f" {value:>15{fmt}}"
            if dnne:
                value = dnne.get(key, 0)
                row += f" {value:>15{fmt}}"
            print(row)
        
        # Calculated metric
        if igenv and dnne:
            igenv_ips = igenv['num_epochs'] / igenv['total_time'] if igenv.get('total_time', 0) > 0 else 0
            dnne_ips = dnne['num_epochs'] / dnne['total_time'] if dnne.get('total_time', 0) > 0 else 0
            print(f"{'Epochs/sec':<30} {igenv_ips:>15.2f} {dnne_ips:>15.2f}")
        
        print("=" * 60)
        
        # Render metrics (if available)
        igenv_render = igenv.get('render_metrics', {}) if igenv else {}
        dnne_render = dnne.get('render_metrics', {}) if dnne else {}
        
        if igenv_render or dnne_render:
            print("\n🎥 RENDER METRICS")
            print("-" * 60)
            
            render_metrics = [
                ('Total Renders', 'total_renders', 'd'),
                ('Renders/sec', 'renders_per_sec', '.1f'),
                ('Total Step Graphics', 'total_step_graphics', 'd'),
                ('Step Graphics/sec', 'step_graphics_per_sec', '.1f'),
                ('Avg Render Time (ms)', 'avg_render_time_ms', '.2f')
            ]
            
            for label, key, fmt in render_metrics:
                row = f"{label:<30}"
                if igenv_render:
                    value = igenv_render.get(key, 0)
                    row += f" {value:>15{fmt}}"
                if dnne_render:
                    value = dnne_render.get(key, 0)
                    row += f" {value:>15{fmt}}"
                print(row)

        # Performance comparison
        if igenv and dnne and igenv.get('steps_per_sec', 0) > 0:
            ratio = dnne.get('steps_per_sec', 0) / igenv.get('steps_per_sec', 0)
            print(f"\nRelative Performance: {ratio:.2f}x")
            
            if ratio > 1.1:
                print(f"✅ DNNE is {ratio:.1f}x faster")
            elif ratio < 0.9:
                print(f"❌ DNNE is {1/ratio:.1f}x slower")
            else:
                print("✅ Performance is comparable")
                
            # Render performance comparison
            if igenv_render and dnne_render:
                igenv_renders_per_sec = igenv_render.get('renders_per_sec', 0)
                dnne_renders_per_sec = dnne_render.get('renders_per_sec', 0)
                
                if igenv_renders_per_sec > 0:
                    render_ratio = dnne_renders_per_sec / igenv_renders_per_sec
                    print(f"Relative Render Performance: {render_ratio:.2f}x")
                    
                    if render_ratio == 0:
                        print("ℹ️  DNNE has no renders (headless mode)")
                    elif render_ratio < 0.8:
                        print(f"⚠️  DNNE renders {1/render_ratio:.1f}x less frequently (may appear slower visually)")
                    elif render_ratio > 1.2:
                        print(f"✅ DNNE renders {render_ratio:.1f}x more frequently")
                    else:
                        print("✅ Render frequency is comparable")
        
        # Learning performance comparison
        self._format_learning_comparison(igenv, dnne)
    
    def _format_detailed_comparison(self, results):
        """Format detailed performance comparison with timing breakdown"""
        print("\n📊 TRAINING PERFORMANCE COMPARISON")
        print("=" * 70)
        
        # Check what results we have
        igenv = results.get('isaacgym', {})
        dnne = results.get('dnne', {})
        
        if not igenv and not dnne:
            print("❌ No results to display")
            return
        
        # Prepare data
        igenv_timings = igenv.get('timings', {}) if igenv else {}
        dnne_timings = dnne.get('timings', {}) if dnne else {}
        
        # Header
        print("=" * 70)
        if igenv and dnne:
            print(f"{'':35} {'IsaacGymEnvs':>17} {'DNNE':>17}")
        elif igenv:
            print(f"{'':35} {'IsaacGymEnvs':>17}")
        else:
            print(f"{'':35} {'DNNE':>17}")
        print("=" * 70)
        
        # Primary metrics
        self._format_metric_row('env.step() per sec', 
                               igenv.get('steps_per_sec', 0) if igenv else None,
                               dnne.get('steps_per_sec', 0) if dnne else None,
                               fmt='.1f')
        
        self._format_metric_row('Total step() calls',
                               igenv.get('step_count', 0) if igenv else None,
                               dnne.get('step_count', 0) if dnne else None,
                               fmt=',d')
        
        print("-" * 70)
        print("Breakdown (avg ms per operation):")
        
        # Environment operations
        self._format_timing_row('  env.step() total', 'env_step_total', igenv_timings, dnne_timings)
        self._format_timing_row('  - gym.simulate()', 'gym_simulate', igenv_timings, dnne_timings)
        self._format_timing_row('  - compute_obs()', 'compute_obs', igenv_timings, dnne_timings)
        self._format_timing_row('  - compute_reward()', 'compute_reward', igenv_timings, dnne_timings)
        self._format_timing_row('  - reset_envs()', 'reset_envs', igenv_timings, dnne_timings)
        print()
        
        # PPO training operations
        # Calculate PPO epoch total if we have the components
        ppo_total_igenv = self._calculate_ppo_total(igenv_timings) if igenv_timings else None
        ppo_total_dnne = self._calculate_ppo_total(dnne_timings) if dnne_timings else None
        
        if ppo_total_igenv is not None or ppo_total_dnne is not None:
            self._format_timing_row('  PPO epoch total', None, 
                                   {'ppo_total': ppo_total_igenv} if ppo_total_igenv else None,
                                   {'ppo_total': ppo_total_dnne} if ppo_total_dnne else None)
        else:
            print("  PPO epoch total                           ???               ???")
            
        self._format_timing_row('  - collect_rollout', 'collect_rollout', igenv_timings, dnne_timings)
        self._format_timing_row('  - compute_returns', 'compute_returns', igenv_timings, dnne_timings)
        
        # Calculate policy update total if we have the components
        policy_total_igenv = self._calculate_policy_total(igenv_timings) if igenv_timings else None
        policy_total_dnne = self._calculate_policy_total(dnne_timings) if dnne_timings else None
        
        if policy_total_igenv is not None or policy_total_dnne is not None:
            self._format_timing_row('  - policy_update', None,
                                   {'policy_total': policy_total_igenv} if policy_total_igenv else None,
                                   {'policy_total': policy_total_dnne} if policy_total_dnne else None)
        else:
            print("  - policy_update                           ???               ???")
            
        self._format_timing_row('    - forward pass', 'policy_forward', igenv_timings, dnne_timings)
        self._format_timing_row('    - backward pass', 'policy_backward', igenv_timings, dnne_timings)
        print()
        
        # Other overheads
        print("  Other overheads")
        self._format_timing_row('  - queue operations', 'queue_operations', 
                               igenv_timings if igenv else None, 
                               dnne_timings if dnne else None,
                               igenv_default='N/A')
        print("  - data transfers                          ???               ???")
        print()
        
        # Summary metrics
        igenv_summary = igenv.get('summary', {}) if igenv else {}
        dnne_summary = dnne.get('summary', {}) if dnne else {}
        
        self._format_metric_row('Init time (one-time):',
                               igenv_summary.get('init_time', 0) if igenv else None,
                               dnne_summary.get('init_time', 0) if dnne else None,
                               fmt='.1f')
        
        self._format_metric_row('Total training time:',
                               igenv.get('total_time', 0) if igenv else None,
                               dnne.get('total_time', 0) if dnne else None,
                               fmt='.2f')
        
        self._format_metric_row('PPO epochs/sec:',
                               igenv_summary.get('epochs_per_sec', 0) if igenv else None,
                               dnne_summary.get('epochs_per_sec', 0) if dnne else None,
                               fmt='.2f')
        
        print("=" * 70)
        
        # Render metrics (if available)
        igenv_render = igenv.get('render_metrics', {}) if igenv else {}
        dnne_render = dnne.get('render_metrics', {}) if dnne else {}
        
        if igenv_render or dnne_render:
            print("\n🎥 RENDER METRICS")
            print("-" * 70)
            
            self._format_metric_row('Total Renders:',
                                   igenv_render.get('total_renders', 0) if igenv_render else None,
                                   dnne_render.get('total_renders', 0) if dnne_render else None,
                                   fmt='d')
            
            self._format_metric_row('Renders/sec:',
                                   igenv_render.get('renders_per_sec', 0) if igenv_render else None,
                                   dnne_render.get('renders_per_sec', 0) if dnne_render else None,
                                   fmt='.1f')
            
            self._format_metric_row('Total Step Graphics:',
                                   igenv_render.get('total_step_graphics', 0) if igenv_render else None,
                                   dnne_render.get('total_step_graphics', 0) if dnne_render else None,
                                   fmt='d')
            
            self._format_metric_row('Step Graphics/sec:',
                                   igenv_render.get('step_graphics_per_sec', 0) if igenv_render else None,
                                   dnne_render.get('step_graphics_per_sec', 0) if dnne_render else None,
                                   fmt='.1f')
            
            if igenv_render.get('avg_render_time_ms', 0) > 0 or dnne_render.get('avg_render_time_ms', 0) > 0:
                self._format_metric_row('Avg Render Time (ms):',
                                       igenv_render.get('avg_render_time_ms', 0) if igenv_render else None,
                                       dnne_render.get('avg_render_time_ms', 0) if dnne_render else None,
                                       fmt='.2f')
        
        # Performance comparison
        if igenv and dnne and igenv.get('steps_per_sec', 0) > 0:
            ratio = dnne.get('steps_per_sec', 0) / igenv.get('steps_per_sec', 0)
            print(f"\nRelative Performance: {ratio:.2f}x")
            
            if ratio > 1.1:
                print(f"✅ DNNE is {ratio:.1f}x faster")
            elif ratio < 0.9:
                print(f"❌ DNNE is {1/ratio:.1f}x slower")
            else:
                print("✅ Performance is comparable")
                
            # Render performance comparison
            if igenv_render and dnne_render:
                igenv_renders_per_sec = igenv_render.get('renders_per_sec', 0)
                dnne_renders_per_sec = dnne_render.get('renders_per_sec', 0)
                
                if igenv_renders_per_sec > 0:
                    render_ratio = dnne_renders_per_sec / igenv_renders_per_sec
                    print(f"Relative Render Performance: {render_ratio:.2f}x")
                    
                    if render_ratio == 0:
                        print("ℹ️  DNNE has no renders (headless mode)")
                    elif render_ratio < 0.8:
                        print(f"⚠️  DNNE renders {1/render_ratio:.1f}x less frequently (may appear slower visually)")
                    elif render_ratio > 1.2:
                        print(f"✅ DNNE renders {render_ratio:.1f}x more frequently")
                    else:
                        print("✅ Render frequency is comparable")
        
        # Learning performance comparison
        self._format_learning_comparison(igenv, dnne)
        
        # Timing coverage info
        if self.mode == 'detailed':
            print("\n📈 Profiling Coverage:")
            if igenv and 'timing_percentages' in igenv:
                total_pct = sum(igenv['timing_percentages'].values())
                print(f"  IsaacGymEnvs: {total_pct:.1f}% of execution time tracked")
            if dnne and 'timing_percentages' in dnne:
                total_pct = sum(dnne['timing_percentages'].values())
                print(f"  DNNE: {total_pct:.1f}% of execution time tracked")
    
    def _format_metric_row(self, label, igenv_value, dnne_value, fmt=''):
        """Format a single metric row"""
        row = f"{label:<35}"
        
        if igenv_value is not None:
            if fmt:
                row += f" {igenv_value:>17{fmt}}"
            else:
                row += f" {igenv_value:>17}"
        elif igenv_value is not None or dnne_value is not None:
            row += " " * 18
            
        if dnne_value is not None:
            if fmt:
                row += f" {dnne_value:>17{fmt}}"
            else:
                row += f" {dnne_value:>17}"
                
        print(row)
    
    def _format_timing_row(self, label, key, igenv_timings, dnne_timings, 
                          igenv_default='???', dnne_default='???'):
        """Format a timing breakdown row"""
        igenv_time = None
        dnne_time = None
        
        if igenv_timings is not None:
            if key is None:
                # Handle special calculated values
                if 'ppo_total' in igenv_timings:
                    igenv_time = f"{igenv_timings['ppo_total']:.2f}"
                elif 'policy_total' in igenv_timings:
                    igenv_time = f"{igenv_timings['policy_total']:.2f}"
            elif key in igenv_timings and igenv_timings[key] is not None:
                igenv_time = f"{igenv_timings[key]:.2f}"
            else:
                igenv_time = igenv_default
                
        if dnne_timings is not None:
            if key is None:
                # Handle special calculated values
                if 'ppo_total' in dnne_timings:
                    dnne_time = f"{dnne_timings['ppo_total']:.2f}"
                elif 'policy_total' in dnne_timings:
                    dnne_time = f"{dnne_timings['policy_total']:.2f}"
            elif key in dnne_timings and dnne_timings[key] is not None:
                dnne_time = f"{dnne_timings[key]:.2f}"
            else:
                dnne_time = dnne_default
        
        row = f"{label:<35}"
        if igenv_time is not None:
            row += f" {igenv_time:>17}"
        elif igenv_timings is not None or dnne_timings is not None:
            row += " " * 18
            
        if dnne_time is not None:
            row += f" {dnne_time:>17}"
            
        print(row)
    
    def _calculate_ppo_total(self, timings):
        """Calculate total PPO epoch time from components"""
        if not timings:
            return None
            
        total = 0
        components = ['collect_rollout', 'compute_returns', 'policy_forward', 'policy_backward']
        found_any = False
        
        for component in components:
            if component in timings and timings[component] is not None:
                total += timings[component]
                found_any = True
                
        return total if found_any else None
    
    def _calculate_policy_total(self, timings):
        """Calculate total policy update time from components"""
        if not timings:
            return None
            
        total = 0
        components = ['policy_forward', 'policy_backward', 'optimizer_step']
        found_any = False
        
        for component in components:
            if component in timings and timings[component] is not None:
                total += timings[component]
                found_any = True
                
        return total if found_any else None
    def _format_learning_comparison(self, igenv, dnne):
        """Format learning performance comparison between systems"""
        igenv_learning = igenv.get("learning_metrics", {}) if igenv else {}
        dnne_learning = dnne.get("learning_metrics", {}) if dnne else {}
        
        # Check if any learning data is available
        igenv_has_data = igenv_learning.get("data_available", False)
        dnne_has_data = dnne_learning.get("data_available", False)
        
        if not igenv_has_data and not dnne_has_data:
            print("\n📚 LEARNING PERFORMANCE")
            print("-" * 60)
            print("ℹ️  No episode return data available from either system")
            return
        
        print("\n📚 LEARNING PERFORMANCE")
        print("-" * 60)
        
        # Header
        if igenv_has_data and dnne_has_data:
            print(f"{'Metric':30} {'IsaacGymEnvs':>15} {'DNNE':>15}")
        elif igenv_has_data:
            print(f"{'Metric':30} {'IsaacGymEnvs':>15}")
        else:
            print(f"{'Metric':30} {'DNNE':>15}")
        
        print("-" * 60)
        
        # Episode metrics
        learning_metrics = [
            ("Total Episodes", "total_episodes", "d"),
            ("Completed Episodes", "completed_episodes", "d"),
            ("Avg Episode Return", "average_episode_return", ".1f"),
            ("Data Source", "source", "s")
        ]
        
        for label, key, fmt in learning_metrics:
            row = f"{label:<30}"
            
            if igenv_has_data:
                value = igenv_learning.get(key, "N/A")
                if fmt == "s":
                    row += f" {str(value):>15}"
                elif value != "N/A":
                    row += f" {value:>15{fmt}}"
                else:
                    row += f" {'N/A':>15}"
            
            if dnne_has_data:
                value = dnne_learning.get(key, "N/A")
                if fmt == "s":
                    row += f" {str(value):>15}"
                elif value != "N/A":
                    row += f" {value:>15{fmt}}"
                else:
                    row += f" {'N/A':>15}"
            
            print(row)
        
        # Learning performance comparison
        if igenv_has_data and dnne_has_data:
            igenv_avg = igenv_learning.get("average_episode_return", 0)
            dnne_avg = dnne_learning.get("average_episode_return", 0)
            
            if igenv_avg > 0:
                learning_ratio = dnne_avg / igenv_avg
                print(f"\nRelative Learning Performance: {learning_ratio:.2f}x")
                
                if 0.8 <= learning_ratio <= 1.2:
                    print("✅ Learning performance is comparable")
                elif learning_ratio > 1.2:
                    print(f"✅ DNNE learns {learning_ratio:.1f}x better episode returns")
                else:
                    print(f"❌ DNNE learns {1/learning_ratio:.1f}x worse episode returns")
            else:
                print("ℹ️  Cannot compare learning - IsaacGymEnvs baseline is zero")
        elif igenv_has_data:
            print("ℹ️  Only IsaacGymEnvs learning data available")
        else:
            print("ℹ️  Only DNNE learning data available")
