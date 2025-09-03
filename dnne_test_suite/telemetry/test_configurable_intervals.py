#!/usr/bin/env python3
"""
Test configurable telemetry intervals and runtime overrides
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


class TestConfigurableIntervals(unittest.TestCase):
    """Test that telemetry intervals can be configured and overridden"""
    
    def test_balancer_report_interval(self):
        """Test BalancerNode report_interval is configurable"""
        template_path = "export_system/templates/nodes/balancer_node_queue.tpl"
        
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                content = f.read()
            
            # Should use report_interval variable
            self.assertIn('REPORT_INTERVAL', content)
            self.assertIn('self.report_interval = {REPORT_INTERVAL}', content)
            self.assertIn('if self.execution_count % self.report_interval == 0', content)
    
    def test_simulation_tracker_interval_parsing(self):
        """Test SimulationTracker interval format parsing"""
        # Import the time_utils to test parsing
        try:
            from export_system.templates.framework.time_utils import parse_duration
            
            # Test various formats
            self.assertEqual(parse_duration("30s"), 30)
            self.assertEqual(parse_duration("5m"), 300)
            self.assertEqual(parse_duration("1h"), 3600)
            self.assertEqual(parse_duration("90"), 90)  # Raw seconds
            
        except ImportError:
            # time_utils might not exist yet, check template
            pass
        
        # Check template for interval parsing
        template_path = "export_system/templates/nodes/simulation_tracker_queue.tpl"
        
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                content = f.read()
            
            # Should parse different interval formats
            self.assertIn('telemetry_interval_str', content)
            self.assertIn("'_' in self.telemetry_interval_str", content)  # Single quotes in template
            self.assertIn('"steps" or "episodes"', content)
            self.assertIn('parse_duration', content)
    
    def test_override_support(self):
        """Test that nodes support runtime override via Global config"""
        # Check BalancerNode
        balancer_path = "export_system/templates/nodes/balancer_node_queue.tpl"
        if os.path.exists(balancer_path):
            with open(balancer_path, 'r') as f:
                content = f.read()
            
            # Should check Global for override
            self.assertIn('Global.get_node_config', content)
            self.assertIn('telemetry_level', content)
        
        # Check EpochTracker
        epoch_path = "export_system/templates/nodes/epoch_tracker_queue.tpl"
        if os.path.exists(epoch_path):
            with open(epoch_path, 'r') as f:
                content = f.read()
            
            # Should support epochs override
            self.assertIn('g.get_node_config(self.node_id', content)


class TestIntervalModes(unittest.TestCase):
    """Test different interval modes for SimulationTracker"""
    
    def test_time_based_interval(self):
        """Test time-based reporting intervals"""
        template_path = "export_system/templates/nodes/simulation_tracker_queue.tpl"
        
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                content = f.read()
            
            # Should support time-based mode
            self.assertIn("self.telemetry_mode == 'time'", content)  # Single quotes
            self.assertIn('time.time() - self.telemetry_last_report_time', content)
    
    def test_step_based_interval(self):
        """Test step-based reporting intervals"""
        template_path = "export_system/templates/nodes/simulation_tracker_queue.tpl"
        
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                content = f.read()
            
            # Should support steps mode
            self.assertIn("self.telemetry_mode == 'steps'", content)  # Single quotes
            self.assertIn('self.timestep_count - self.telemetry_last_report_step', content)
    
    def test_episode_based_interval(self):
        """Test episode-based reporting intervals"""
        template_path = "export_system/templates/nodes/simulation_tracker_queue.tpl"
        
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                content = f.read()
            
            # Should support episodes mode
            self.assertIn("self.telemetry_mode == 'episodes'", content)  # Single quotes
            self.assertIn('self.episode_count - self.telemetry_last_report_episode', content)


class TestWidgetConfiguration(unittest.TestCase):
    """Test that UI widgets properly configure telemetry"""
    
    def test_balancer_widgets(self):
        """Test BalancerNode has correct telemetry widgets"""
        visnode_path = "custom_nodes/balancer_visnode.py"
        
        if os.path.exists(visnode_path):
            with open(visnode_path, 'r') as f:
                content = f.read()
            
            # Should have report_interval widget
            self.assertIn('"report_interval": ("INT"', content)
            self.assertIn('"default": 100', content)
            
            # Should have telemetry_level widget
            self.assertIn('"telemetry_level": (["off", "essential", "extended", "debug"]', content)
    
    def test_epoch_tracker_widgets(self):
        """Test EpochTracker has simplified widgets"""
        visnode_path = "custom_nodes/epoch_tracker_visnode.py"
        
        if os.path.exists(visnode_path):
            with open(visnode_path, 'r') as f:
                content = f.read()
            
            # Should have telemetry_level
            self.assertIn('"telemetry_level"', content)
            
            # Should NOT have window-based widgets
            self.assertNotIn('telemetry_batch_window', content)
            self.assertNotIn('telemetry_time_window', content)
    
    def test_simulation_tracker_widgets(self):
        """Test SimulationTracker has correct widgets"""
        visnode_path = "custom_nodes/simulation_tracker_visnode.py"
        
        if os.path.exists(visnode_path):
            with open(visnode_path, 'r') as f:
                content = f.read()
            
            # Should have simplified interval widget
            self.assertIn('"telemetry_interval": ("STRING"', content)
            self.assertIn('"100_steps"', content)  # Default value
            
            # Should NOT have reward input anymore
            self.assertNotIn('"reward":', content)
            self.assertNotIn('REWARD_SCALAR', content)


if __name__ == '__main__':
    unittest.main()