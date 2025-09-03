#!/usr/bin/env python3
"""
Test telemetry levels functionality: off, essential, extended, debug
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


class TestTelemetryLevels(unittest.TestCase):
    """Test different telemetry levels work correctly"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the socket module to prevent actual UDP sends
        self.socket_patcher = patch('socket.socket')
        self.mock_socket_class = self.socket_patcher.start()
        self.mock_socket = MagicMock()
        self.mock_socket_class.return_value = self.mock_socket
        
        # Import telemetry after patching socket
        from export_system.templates.framework import telemetry
        self.telemetry_module = telemetry
        
    def tearDown(self):
        """Clean up after tests"""
        self.socket_patcher.stop()
    
    def test_telemetry_off(self):
        """Test that telemetry disabled means no UDP packets sent"""
        client = self.telemetry_module.TelemetryClient(enabled=False)
        
        # Try various reporting methods
        client.report_custom("node_1", "metric1", 100.0)
        client.report_throughput("node_1", 50.0)
        client.report_latency("node_1", 10.5)
        client.report_violation("node_1", "test_violation", 100.0, 90.0)
        
        # Verify no sendto calls were made
        self.mock_socket.sendto.assert_not_called()
    
    def test_telemetry_enabled(self):
        """Test that telemetry enabled sends UDP packets"""
        client = self.telemetry_module.TelemetryClient(enabled=True)
        
        # Report a custom metric
        client.report_custom("node_1", "test_metric", 42.0)
        
        # Verify sendto was called
        self.mock_socket.sendto.assert_called()
        
        # Check the packet format
        call_args = self.mock_socket.sendto.call_args
        packet_data = call_args[0][0]
        
        # Should be pipe-delimited for simple metrics
        self.assertIn(b"test_metric|node_1|42.0|", packet_data)
    
    def test_violation_rate_limiting(self):
        """Test that violations are rate-limited correctly"""
        client = self.telemetry_module.TelemetryClient(
            enabled=True, 
            violation_rate_limit=5  # 5 per second
        )
        
        # Send 10 violations quickly
        for i in range(10):
            client.report_violation("node_1", "test_violation", 100.0, float(i))
        
        # Should have sent 5 (rate limited)
        # Note: violations use JSON format, so check for that
        calls = self.mock_socket.sendto.call_args_list
        violation_calls = [c for c in calls if b'"type": "violation"' in c[0][0]]
        self.assertEqual(len(violation_calls), 5)
    
    def test_enhanced_api_methods(self):
        """Test the new enhanced API methods"""
        client = self.telemetry_module.TelemetryClient(enabled=True)
        
        # Test start_window (should be no-op for UDP)
        client.start_window("node_1", "epoch")
        # No assertion needed - just shouldn't crash
        
        # Test end_window
        stats = {
            "loss_mean": 0.5,
            "accuracy_mean": 0.95,
            "batches": 100
        }
        client.end_window("node_1", stats)
        
        # Should have sent 3 metrics
        self.assertEqual(self.mock_socket.sendto.call_count, 3)
        
        # Test report_metric (should forward to report_custom)
        self.mock_socket.sendto.reset_mock()
        client.report_metric("node_1", "test_metric", 123.0, aggregate=True)
        self.mock_socket.sendto.assert_called_once()
    
    def test_queue_depth_reporting(self):
        """Test queue depth reporting uses correct JSON format"""
        client = self.telemetry_module.TelemetryClient(enabled=True)
        
        client.report_queue_depth("node_1", "input_queue", 5)
        
        call_args = self.mock_socket.sendto.call_args
        packet_data = call_args[0][0]
        
        # Should be JSON format for queue metrics
        import json
        data = json.loads(packet_data)
        self.assertEqual(data["type"], "queue")
        self.assertEqual(data["node_id"], "node_1")
        self.assertEqual(data["queue"], "input_queue")
        self.assertEqual(data["depth"], 5)


class TestNodeTelemetryIntegration(unittest.TestCase):
    """Test how nodes integrate with telemetry at different levels"""
    
    def test_balancer_node_telemetry_levels(self):
        """Test BalancerNode respects telemetry levels"""
        # This would test the actual template behavior
        # For now, just verify the template variables exist
        template_path = "export_system/templates/nodes/balancer_node_queue.tpl"
        
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                content = f.read()
            
            # Check for telemetry level checks
            self.assertIn('telemetry_level', content)
            self.assertIn('"essential"', content)
            self.assertIn('"extended"', content) 
            self.assertIn('"debug"', content)
            self.assertIn('report_interval', content)
    
    def test_epoch_tracker_telemetry(self):
        """Test EpochTracker only reports at epoch completion"""
        template_path = "export_system/templates/nodes/epoch_tracker_queue.tpl"
        
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                content = f.read()
            
            # Should only report when epoch completed
            self.assertIn('if epoch_stats.get("completed"', content)
            self.assertIn('telemetry_level != "off"', content)
            
            # Should report essential metrics
            self.assertIn('telemetry.report_custom(self.node_id, "epoch"', content)
            self.assertIn('telemetry.report_custom(self.node_id, "loss_mean"', content)
    
    def test_simulation_tracker_loss_focus(self):
        """Test SimulationTracker focuses on loss, not rewards"""
        template_path = "export_system/templates/nodes/simulation_tracker_queue.tpl"
        
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                content = f.read()
            
            # Should NOT have reward buffers or reward reporting
            self.assertNotIn('telemetry_reward_buffer', content)
            self.assertNotIn('reward_mean', content)
            self.assertNotIn('reward_samples', content)
            
            # Should have loss tracking
            self.assertIn('telemetry_loss_buffer', content)
            self.assertIn('episode_losses', content)
            self.assertIn('best_loss', content)
            
            # Control metrics should use loss
            self.assertIn('avg_loss', content)
            self.assertNotIn('avg_reward', content)


if __name__ == '__main__':
    unittest.main()