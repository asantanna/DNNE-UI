#!/usr/bin/env python3
"""
Simple test of SimulationTracker telemetry configuration without full imports.
"""

import sys
import os
sys.path.append('/home/asantanna/DNNE/DNNE-UI')


def test_template_variables():
    """Test that template variables are correctly prepared."""
    print("\n=== Testing Template Variable Preparation ===\n")
    
    # Simulate the prepare_template_vars logic
    test_configs = [
        {
            "name": "Time-based with stats (10s interval)",
            "widgets": [1000, 0.95, "time", "10s", True],
            "expected": {
                "mode": "time",
                "interval": "10s",
                "stats": True
            }
        },
        {
            "name": "Step-based without stats (100 steps)",
            "widgets": [500, 0.9, "steps", "100", False],
            "expected": {
                "mode": "steps",
                "interval": "100",
                "stats": False
            }
        },
        {
            "name": "Episode-based with stats (5 episodes)",
            "widgets": [2000, 0.99, "episodes", "5", True],
            "expected": {
                "mode": "episodes",
                "interval": "5",
                "stats": True
            }
        },
        {
            "name": "Complex time interval (2m30s)",
            "widgets": [1000, 0.95, "time", "2m30s", True],
            "expected": {
                "mode": "time",
                "interval": "2m30s",
                "stats": True
            }
        },
    ]
    
    for config in test_configs:
        print(f"  {config['name']}:")
        widgets = config["widgets"]
        
        # Extract values as the exporter would
        telemetry_mode = widgets[2] if len(widgets) > 2 else "time"
        telemetry_interval = widgets[3] if len(widgets) > 3 else "10s"
        telemetry_stats = widgets[4] if len(widgets) > 4 else True
        
        # Check values
        expected = config["expected"]
        
        mode_ok = telemetry_mode == expected["mode"]
        interval_ok = telemetry_interval == expected["interval"]
        stats_ok = telemetry_stats == expected["stats"]
        
        print(f"    Mode:     {telemetry_mode:10} {'✓' if mode_ok else '✗'}")
        print(f"    Interval: {telemetry_interval:10} {'✓' if interval_ok else '✗'}")
        print(f"    Stats:    {telemetry_stats!s:10} {'✓' if stats_ok else '✗'}")
        
        if mode_ok and interval_ok and stats_ok:
            print(f"    Overall: ✓ All correct")
        else:
            print(f"    Overall: ✗ Some errors")
        print()


def test_telemetry_logic():
    """Test the telemetry reporting logic."""
    print("\n=== Testing Telemetry Reporting Logic ===\n")
    
    from export_system.templates.framework.time_utils import parse_duration
    
    # Simulate different reporting scenarios
    scenarios = [
        {
            "name": "Time-based: 10s interval, 15s elapsed",
            "mode": "time",
            "interval": "10s",
            "elapsed_time": 15,
            "elapsed_steps": 150,
            "elapsed_episodes": 2,
            "should_report": True
        },
        {
            "name": "Time-based: 10s interval, 5s elapsed",
            "mode": "time",
            "interval": "10s",
            "elapsed_time": 5,
            "elapsed_steps": 50,
            "elapsed_episodes": 0,
            "should_report": False
        },
        {
            "name": "Step-based: 100 steps interval, 150 steps elapsed",
            "mode": "steps",
            "interval": "100",
            "elapsed_time": 10,
            "elapsed_steps": 150,
            "elapsed_episodes": 1,
            "should_report": True
        },
        {
            "name": "Step-based: 100 steps interval, 50 steps elapsed",
            "mode": "steps",
            "interval": "100",
            "elapsed_time": 5,
            "elapsed_steps": 50,
            "elapsed_episodes": 0,
            "should_report": False
        },
        {
            "name": "Episode-based: 5 episodes interval, 5 episodes elapsed",
            "mode": "episodes",
            "interval": "5",
            "elapsed_time": 60,
            "elapsed_steps": 500,
            "elapsed_episodes": 5,
            "should_report": True
        },
        {
            "name": "Episode-based: 5 episodes interval, 3 episodes elapsed",
            "mode": "episodes",
            "interval": "5",
            "elapsed_time": 30,
            "elapsed_steps": 300,
            "elapsed_episodes": 3,
            "should_report": False
        },
    ]
    
    for scenario in scenarios:
        print(f"  {scenario['name']}:")
        
        # Simulate the _should_report_telemetry logic
        mode = scenario["mode"]
        interval_str = scenario["interval"]
        
        should_report = False
        if mode == "time":
            interval = parse_duration(interval_str)
            should_report = scenario["elapsed_time"] >= interval
        elif mode == "steps":
            interval = int(interval_str)
            should_report = scenario["elapsed_steps"] >= interval
        elif mode == "episodes":
            interval = int(interval_str)
            should_report = scenario["elapsed_episodes"] >= interval
        
        expected = scenario["should_report"]
        status = "✓" if should_report == expected else "✗"
        
        print(f"    Should report: {should_report} (expected: {expected}) {status}")
        print()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SimulationTracker Telemetry Configuration Test")
    print("="*60)
    
    test_template_variables()
    test_telemetry_logic()
    
    print("="*60)
    print("Test Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()