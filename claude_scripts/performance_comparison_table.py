#!/usr/bin/env python3
"""
Performance Comparison Table: DNNE vs IsaacGymEnvs

This script runs both DNNE and IsaacGymEnvs implementations (or uses baseline data)
and generates a formatted comparison table showing key performance metrics.
"""

import os
import sys
import time
import re
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

class PerformanceMetrics:
    """Container for performance metrics"""
    def __init__(self):
        self.init_time: Optional[float] = None
        self.avg_fps: Optional[float] = None
        self.peak_fps: Optional[float] = None
        self.batch_size: Optional[int] = None
        self.forward_pass_ms: Optional[float] = None
        self.training_speed: Optional[float] = None  # epochs per minute
        self.status: str = "UNKNOWN"
        self.notes: list = []
        self.total_steps: Optional[int] = None
        self.compute_nodes: Dict[str, int] = {}

def parse_dnne_config(export_dir: Path) -> Dict[str, Any]:
    """Parse configuration values directly from exported source code"""
    config = {}
    
    # Parse IsaacGymEnvNode for num_envs (batch size)
    isaac_gym_node = export_dir / "nodes" / "isaacgymenvnode_7.py"
    if isaac_gym_node.exists():
        with open(isaac_gym_node, 'r') as f:
            content = f.read()
            
        # Parse num_envs (batch size)
        match = re.search(r'self\.num_envs\s*=\s*(\d+)', content)
        if match:
            config['num_envs'] = int(match.group(1))
            
        # Parse device
        match = re.search(r'self\.device\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            config['device'] = match.group(1)
            
        # Parse env_name
        match = re.search(r'self\.env_name\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            config['env_name'] = match.group(1)
    
    # Parse PPOTrainerNode for training parameters
    ppo_trainer_node = export_dir / "nodes" / "ppotrainernode_6.py"
    if ppo_trainer_node.exists():
        with open(ppo_trainer_node, 'r') as f:
            content = f.read()
            
        # Parse horizon_length
        match = re.search(r'self\.horizon_length\s*=\s*(\d+)', content)
        if match:
            config['horizon_length'] = int(match.group(1))
            
        # Parse minibatch_size
        match = re.search(r'self\.minibatch_size\s*=\s*(\d+)', content)
        if match:
            config['minibatch_size'] = int(match.group(1))
            
        # Parse learning_rate
        match = re.search(r'self\.learning_rate\s*=\s*([\d.]+)', content)
        if match:
            config['learning_rate'] = float(match.group(1))
    
    # Parse PPOAgentNode for additional parameters
    ppo_agent_node = export_dir / "nodes" / "ppoagentnode_3.py"
    if ppo_agent_node.exists():
        with open(ppo_agent_node, 'r') as f:
            content = f.read()
            
        # Parse learning_rate (backup if not found in trainer)
        if 'learning_rate' not in config:
            match = re.search(r'self\.learning_rate\s*=\s*([\d.]+)', content)
            if match:
                config['learning_rate'] = float(match.group(1))
    
    return config

def run_dnne_test(test_duration: int = 15, timeout_seconds: int = 60) -> PerformanceMetrics:
    """Run DNNE Cartpole test and collect metrics"""
    metrics = PerformanceMetrics()
    
    print("🔧 Running DNNE Cartpole test...")
    
    # Change to export directory
    export_dir = Path("/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO")
    if not export_dir.exists():
        metrics.status = "NOT_FOUND"
        metrics.notes.append("Export directory not found")
        return metrics
    
    # Parse configuration from source code
    config = parse_dnne_config(export_dir)
    print(f"  📋 Parsed config: {config}")
    
    # Use parsed batch size (num_envs) instead of hardcoded value
    if 'num_envs' in config:
        metrics.batch_size = config['num_envs']
        print(f"  📊 Batch size from config: {metrics.batch_size}")
    else:
        metrics.batch_size = 512  # Fallback
        metrics.notes.append("Could not parse batch size from config, using fallback")
    
    # Prepare environment activation
    conda_activate = "source /home/asantanna/miniconda/bin/activate DNNE_PY38"
    
    # Run with timeout and profile mode to capture performance data
    cmd = f"{conda_activate} && cd {export_dir} && python runner.py --headless --timeout {test_duration}s --profile"
    
    print(f"  Running command: {cmd}")
    
    start_time = time.time()
    try:
        # Use subprocess.Popen for better timeout handling with output capture
        process = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            elapsed = time.time() - start_time
            metrics.status = "SUCCESS"
            metrics.init_time = elapsed
        except subprocess.TimeoutExpired:
            # Kill the process and get partial output
            process.kill()
            stdout, stderr = process.communicate()
            elapsed = time.time() - start_time
            metrics.status = "TIMEOUT"
            metrics.init_time = elapsed
            metrics.notes.append(f"Timeout after {elapsed:.1f}s - captured partial output")
        
    except Exception as e:
        metrics.status = "ERROR"
        metrics.notes.append(f"Exception: {str(e)}")
        return metrics
        
    # Parse output
    all_output = stdout + "\n" + stderr
    
    # Parse metrics from output
    or_node_outputs = 0
    training_steps = 0
    actual_runtime = elapsed
    
    for line in all_output.split('\n'):
        # Look for profiling timing output
        if 'PPOAgent-' in line and 'Forward_Pass:' in line:
            # Parse: PPOAgent-3 (Forward_Pass): 5.234ms (t+123.456ms) | Model_features: 1.2ms | Value_computation: 0.8ms | Policy_computation: 2.1ms | Batch_size: 512
            match = re.search(r'PPOAgent-\d+.*?Forward_Pass.*?:\s*([\d.]+)ms', line)
            if match:
                metrics.forward_pass_ms = float(match.group(1))
        
        # Look for actual runtime from completion message
        elif 'Completed' in line and 'run in' in line and 'seconds' in line:
            # Parse: "✅ Completed 15s run in 15.1 seconds"
            match = re.search(r'run in ([\d.]+) seconds', line)
            if match:
                actual_runtime = float(match.group(1))
        
        # Count OR Node outputs (data flow indicator) - legacy parsing
        elif 'OR Node: Routing' in line and 'output #' in line:
            or_node_outputs += 1
            
        # Count PPO training steps - legacy parsing
        elif 'PPO training step' in line and 'complete' in line:
            training_steps += 1
        
        # Look for final statistics computations
        elif 'computations, avg time:' in line:
            # Parse: "  2: 124 computations, avg time: 0.000s"
            parts = line.split()
            if len(parts) >= 2:
                node_id = parts[0].rstrip(':')
                try:
                    comp_count = int(parts[1])
                    metrics.compute_nodes[node_id] = comp_count
                    
                    # Extract avg compute time for this node
                    time_match = re.search(r'avg time:\s*([\d.]+)s', line)
                    if time_match and node_id == "3":  # PPO Agent node
                        avg_time_s = float(time_match.group(1))
                        if metrics.forward_pass_ms is None and avg_time_s > 0:
                            metrics.forward_pass_ms = avg_time_s * 1000  # Convert to ms
                except ValueError:
                    pass
    
    # Calculate performance metrics using new computation-based approach
    if metrics.compute_nodes:
        # Use OR node (node "2") as the primary throughput indicator
        if "2" in metrics.compute_nodes:
            or_computations = metrics.compute_nodes["2"]
            metrics.total_steps = or_computations
            if actual_runtime > 0:
                metrics.avg_fps = or_computations / actual_runtime
                
        # Use PPO Agent node (node "3") computations as secondary indicator
        elif "3" in metrics.compute_nodes:
            ppo_computations = metrics.compute_nodes["3"]
            metrics.total_steps = ppo_computations
            if actual_runtime > 0:
                metrics.avg_fps = ppo_computations / actual_runtime
                
        # Fallback to total computations across all nodes
        else:
            metrics.total_steps = sum(metrics.compute_nodes.values())
            if actual_runtime > 0:
                metrics.avg_fps = metrics.total_steps / actual_runtime
    
    # Legacy calculation for backward compatibility
    elif or_node_outputs > 0:
        metrics.total_steps = or_node_outputs
        if actual_runtime > 0:
            metrics.avg_fps = or_node_outputs / actual_runtime
            
    # Update status based on successful computations
    if metrics.compute_nodes and any(count > 0 for count in metrics.compute_nodes.values()):
        metrics.status = "SUCCESS"
        total_computations = sum(metrics.compute_nodes.values())
        metrics.notes.append(f"Training active: {total_computations} total computations across all nodes")
        
        # Add detailed node breakdown
        node_details = ", ".join([f"Node {nid}: {count}" for nid, count in metrics.compute_nodes.items() if count > 0])
        metrics.notes.append(f"Computation breakdown: {node_details}")
        
    # Legacy status update
    elif training_steps > 0 and or_node_outputs > 0:
        metrics.status = "SUCCESS"
        metrics.notes.append(f"Training active: {training_steps} PPO steps, {or_node_outputs} data flows")
    elif metrics.status == "TIMEOUT" and or_node_outputs > 0:
        metrics.status = "PARTIAL_SUCCESS" 
        metrics.notes.append(f"Training started but incomplete: {or_node_outputs} data flows")
    
    # Add initialization hang note if detected
    if "Hang detected" in all_output or "environment factory" in all_output:
        metrics.notes.append("Initialization hangs in environment factory")
    
    # Look for specific initialization messages
    if "PhysX Engine" in all_output:
        metrics.notes.append("Isaac Gym PhysX initialized successfully")
    if "Creating Cartpole environment" in all_output:
        metrics.notes.append("Environment creation started but did not complete")
    
    return metrics

def get_isaacgymenvs_baseline() -> PerformanceMetrics:
    """Get IsaacGymEnvs baseline metrics from our performance analysis"""
    metrics = PerformanceMetrics()
    
    # Data from performance analysis documents
    metrics.init_time = 5.2
    metrics.avg_fps = 32000  # Conservative average
    metrics.peak_fps = 36897
    metrics.batch_size = 512
    metrics.forward_pass_ms = 0.8  # Estimated from FPS
    metrics.training_speed = 68  # epochs per minute (34 epochs in 30s)
    metrics.status = "SUCCESS"
    metrics.notes.append("Baseline data from performance analysis")
    
    return metrics

async def run_isaacgymenvs_test(timeout_seconds: int = 60) -> PerformanceMetrics:
    """Optionally run actual IsaacGymEnvs test"""
    metrics = PerformanceMetrics()
    
    print("🔧 Running IsaacGymEnvs Cartpole test...")
    
    # Check if IsaacGymEnvs is available
    ige_path = Path("/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs")
    if not ige_path.exists():
        print("⚠️  IsaacGymEnvs not found, using baseline data")
        return get_isaacgymenvs_baseline()
    
    # For now, just use baseline data
    # Full implementation would run: python isaacgymenvs/train.py task=Cartpole headless=True
    return get_isaacgymenvs_baseline()

def format_value(value: Optional[float], suffix: str = "", decimals: int = 1, special_case: str = None) -> str:
    """Format a metric value for display"""
    if special_case:
        return special_case
    if value is None:
        return "N/A"
    if suffix == "ms" and decimals == 1:
        return f"{value:.1f}{suffix}"
    elif suffix == "" and value > 1000:
        return f"{value:,.0f}"
    else:
        return f"{value:.{decimals}f}{suffix}"

def print_comparison_table(dnne: PerformanceMetrics, ige: PerformanceMetrics):
    """Print the formatted comparison table"""
    
    # Calculate column widths
    col1_width = 20
    col2_width = 17
    col3_width = 18
    total_width = col1_width + col2_width + col3_width + 4  # +4 for separators
    
    # Header
    print("\n" + "=" * total_width)
    print("Performance Comparison: DNNE vs IsaacGymEnvs (Cartpole PPO)")
    print("=" * total_width)
    
    # Column headers
    print(f"{'Measurement':<{col1_width}} | {'DNNE':^{col2_width}} | {'IsaacGymEnvs':^{col3_width}}")
    print("-" * col1_width + "-+-" + "-" * col2_width + "-+-" + "-" * col3_width)
    
    # Metrics rows
    dnne_init_time = format_value(dnne.init_time, "s", special_case="Timeout" if dnne.status == "TIMEOUT" else None)
    rows = [
        ("Init Time (s)", dnne_init_time, format_value(ige.init_time, "s")),
        ("Avg FPS", format_value(dnne.avg_fps), format_value(ige.avg_fps)),
        ("Peak FPS", format_value(dnne.peak_fps), format_value(ige.peak_fps)),
        ("Batch Size", format_value(dnne.batch_size), format_value(ige.batch_size)),
        ("Forward Pass (ms)", format_value(dnne.forward_pass_ms, "ms"), format_value(ige.forward_pass_ms, "ms")),
        ("Training Speed", format_value(dnne.training_speed, " eps/min"), format_value(ige.training_speed, " eps/min")),
        ("Status", dnne.status, ige.status),
    ]
    
    for metric, dnne_val, ige_val in rows:
        print(f"{metric:<{col1_width}} | {dnne_val:^{col2_width}} | {ige_val:^{col3_width}}")
    
    print("-" * col1_width + "-+-" + "-" * col2_width + "-+-" + "-" * col3_width)
    
    # Performance analysis
    if dnne.forward_pass_ms and ige.forward_pass_ms:
        overhead = dnne.forward_pass_ms / ige.forward_pass_ms
        print(f"\n📊 Forward Pass Overhead: DNNE is {overhead:.1f}x slower than IsaacGymEnvs")
    
    if dnne.avg_fps and ige.avg_fps:
        fps_ratio = (dnne.avg_fps / ige.avg_fps) * 100
        print(f"📊 Throughput: DNNE achieves {fps_ratio:.3f}% of IsaacGymEnvs performance")
        
        # Calculate speed difference
        speed_ratio = ige.avg_fps / dnne.avg_fps
        print(f"📊 Speed Difference: IsaacGymEnvs is {speed_ratio:,.0f}x faster than DNNE")
    
    # Notes section
    print("\n📝 Notes:")
    if dnne.notes:
        print("  DNNE:")
        for note in dnne.notes:
            print(f"    - {note}")
    if ige.notes:
        print("  IsaacGymEnvs:")
        for note in ige.notes:
            print(f"    - {note}")
    
    # Architecture comparison
    print("\n🏗️  Architecture Differences:")
    print("  - DNNE: Queue-based async node graph execution")
    print("  - IsaacGymEnvs: Direct vectorized execution model")
    print("  - Both use GPU PhysX acceleration for physics simulation")
    
    # Recommendations
    print("\n💡 Optimization Opportunities:")
    if dnne.forward_pass_ms and dnne.forward_pass_ms > 10:
        print(f"  - Optimize forward pass timing ({dnne.forward_pass_ms:.1f}ms is significantly slower than baseline)")
    if dnne.status == "TIMEOUT":
        print("  - Fix environment factory initialization hang")
    if not dnne.avg_fps:
        print("  - Complete initialization to enable performance measurement")
    if dnne.avg_fps and ige.avg_fps and dnne.avg_fps < ige.avg_fps / 100:
        print("  - Address async queue overhead that is causing major performance degradation")

async def main():
    """Main entry point"""
    print("🚀 DNNE vs IsaacGymEnvs Performance Comparison")
    print("  Collecting metrics...")
    
    # Run tests  
    dnne_metrics = run_dnne_test(test_duration=15, timeout_seconds=60)  # Short test, long timeout for init
    ige_metrics = await run_isaacgymenvs_test()
    
    # Print comparison table
    print_comparison_table(dnne_metrics, ige_metrics)
    
    # Return exit code based on DNNE status
    return 0 if dnne_metrics.status == "SUCCESS" else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)