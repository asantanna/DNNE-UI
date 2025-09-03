#!/usr/bin/env python3
"""
Test that telemetry produces simplified, essential metrics only
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


class TestSimplifiedMetrics(unittest.TestCase):
    """Test that metrics have been simplified according to policy"""
    
    def test_no_metrics_logger(self):
        """Test that MetricsLogger has been completely removed"""
        # Check that MetricsLogger file doesn't exist
        metrics_logger_path = "export_system/templates/framework/metrics_logger.py"
        self.assertFalse(os.path.exists(metrics_logger_path), 
                        "MetricsLogger should be deleted")
        
        # Check templates don't reference MetricsLogger
        template_dir = "export_system/templates/nodes"
        if os.path.exists(template_dir):
            for filename in os.listdir(template_dir):
                if filename.endswith('.tpl'):
                    filepath = os.path.join(template_dir, filename)
                    with open(filepath, 'r') as f:
                        content = f.read()
                    self.assertNotIn('MetricsLogger', content, 
                                   f"{filename} should not reference MetricsLogger")
    
    def test_balancer_essential_metrics(self):
        """Test BalancerNode reports only essential metrics"""
        template_path = "export_system/templates/nodes/balancer_node_queue.tpl"
        
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                content = f.read()
            
            # Essential metrics that should be present
            self.assertIn('frequency_current', content)
            self.assertIn('latency_avg', content)
            self.assertIn('violation_count', content)
            
            # Should have level-based reporting
            self.assertIn('if telemetry_level != "off"', content)
            self.assertIn('if telemetry_level in ["extended", "debug"]', content)
    
    def test_epoch_tracker_reduced_metrics(self):
        """Test EpochTracker has reduced metric set"""
        template_path = "export_system/templates/nodes/epoch_tracker_queue.tpl"
        
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                content = f.read()
            
            # Essential metrics only
            self.assertIn('"epoch"', content)
            self.assertIn('"loss_mean"', content)
            self.assertIn('"accuracy_mean"', content)
            
            # Should NOT have percentiles in essential mode
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'p25' in line or 'p50' in line or 'p75' in line:
                    # Check if it's behind a debug/extended check
                    found_guard = False
                    for j in range(max(0, i-5), i):
                        if 'debug' in lines[j] or 'extended' in lines[j]:
                            found_guard = True
                            break
                    if 'p25' in line:  # Percentiles were removed entirely
                        self.fail("Percentiles should be removed from EpochTracker")
    
    def test_simulation_tracker_loss_focused(self):
        """Test SimulationTracker focuses on loss metrics"""
        template_path = "export_system/templates/nodes/simulation_tracker_queue.tpl"
        
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                content = f.read()
            
            # Essential loss metrics
            self.assertIn('"episodes_completed"', content)
            self.assertIn('"loss_mean"', content)
            self.assertIn('"timesteps_total"', content)
            
            # No reward metrics
            self.assertNotIn('"reward_mean"', content)
            self.assertNotIn('"reward_samples"', content)
            self.assertNotIn('"episode_reward', content)


class TestDataVolumeReduction(unittest.TestCase):
    """Test that telemetry changes reduce data volume"""
    
    def test_no_window_reporting_epoch_tracker(self):
        """Test EpochTracker doesn't report during epoch (window-based)"""
        template_path = "export_system/templates/nodes/epoch_tracker_queue.tpl"
        
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                content = f.read()
            
            # Should only report when epoch completed
            self.assertIn('if epoch_stats.get("completed"', content)
            
            # Should NOT have window-based reporting
            self.assertNotIn('_report_window_telemetry', content)
            self.assertNotIn('telemetry_batch_window', content)
            self.assertNotIn('telemetry_time_window', content)
    
    def test_configurable_reporting_frequency(self):
        """Test all nodes have configurable reporting frequency"""
        # BalancerNode
        balancer_path = "export_system/templates/nodes/balancer_node_queue.tpl"
        if os.path.exists(balancer_path):
            with open(balancer_path, 'r') as f:
                content = f.read()
            self.assertIn('report_interval', content)
        
        # SimulationTracker  
        sim_path = "export_system/templates/nodes/simulation_tracker_queue.tpl"
        if os.path.exists(sim_path):
            with open(sim_path, 'r') as f:
                content = f.read()
            self.assertIn('telemetry_interval', content)
    
    def test_rate_limiting(self):
        """Test violations are rate-limited"""
        try:
            from export_system.templates.framework.telemetry import SimpleRateLimiter
            
            limiter = SimpleRateLimiter(max_msgs_per_sec=5)
            
            # First 5 should pass
            for i in range(5):
                self.assertTrue(limiter.should_send())
            
            # 6th should fail (within same second)
            self.assertFalse(limiter.should_send())
            
        except ImportError:
            # Check the template at least
            telemetry_path = "export_system/templates/framework/telemetry.py"
            if os.path.exists(telemetry_path):
                with open(telemetry_path, 'r') as f:
                    content = f.read()
                self.assertIn('SimpleRateLimiter', content)
                self.assertIn('violation_rate_limiter', content)


class TestTelemetryPolicy(unittest.TestCase):
    """Test implementation matches telemetry policy document"""
    
    def test_minimal_overhead_principle(self):
        """Test telemetry designed for <0.1% CPU overhead"""
        telemetry_path = "export_system/templates/framework/telemetry.py"
        
        if os.path.exists(telemetry_path):
            with open(telemetry_path, 'r') as f:
                content = f.read()
            
            # Should use non-blocking UDP
            self.assertIn('SOCK_DGRAM', content)
            self.assertIn('setblocking(False)', content)
            
            # Should be fire-and-forget (no waiting for responses)
            self.assertNotIn('recv', content)
            self.assertNotIn('recvfrom', content)
    
    def test_single_source_of_truth(self):
        """Test each metric has one authoritative source"""
        # Check that we don't have duplicate reporting
        
        # BalancerNode should not report to both TelemetryClient and MetricsLogger
        balancer_path = "export_system/templates/nodes/balancer_node_queue.tpl"
        if os.path.exists(balancer_path):
            with open(balancer_path, 'r') as f:
                content = f.read()
            
            # Should only use telemetry client
            self.assertIn('telemetry.report', content)
            self.assertNotIn('MetricsLogger', content)
            self.assertNotIn('logger.record_metric', content)
    
    def test_clear_purpose(self):
        """Test metrics have clear documented purpose"""
        policy_path = "dnne_docs/architecture/telemetry_policy.md"
        
        if os.path.exists(policy_path):
            with open(policy_path, 'r') as f:
                content = f.read()
            
            # Policy should define what each metric is for
            self.assertIn('Essential Metrics', content)
            self.assertIn('Optional Metrics', content)
            self.assertIn('Purpose:', content)


if __name__ == '__main__':
    unittest.main()