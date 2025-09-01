#!/usr/bin/env python3
"""
Test the new SimulationTracker telemetry reporting system.
Tests time parsing utility and telemetry configuration.
"""

import sys
import os
sys.path.append('/home/asantanna/DNNE/DNNE-UI')

from export_system.templates.framework.time_utils import parse_duration, format_duration


def test_time_parsing():
    """Test the time parsing utility function."""
    print("\n=== Testing Time Parsing Utility ===\n")
    
    test_cases = [
        # (input, expected_seconds)
        (30, 30.0),
        (45.5, 45.5),
        ("60", 60.0),
        ("30s", 30.0),
        ("5m", 300.0),
        ("2h", 7200.0),
        ("1h30m", 5400.0),
        ("2m30s", 150.0),
        ("1h5m10s", 3910.0),
        ("90s", 90.0),
        ("0.5s", 0.5),
    ]
    
    for input_val, expected in test_cases:
        try:
            result = parse_duration(input_val)
            status = "✓" if result == expected else f"✗ (got {result})"
            print(f"  parse_duration({repr(input_val):15}) -> {result:8.1f}s  {status}")
        except Exception as e:
            print(f"  parse_duration({repr(input_val):15}) -> ERROR: {e}")
    
    # Test error cases
    print("\n  Testing error cases:")
    error_cases = ["", "invalid", "10x", "-5s", "-30", "h", "m", "s"]
    for input_val in error_cases:
        try:
            result = parse_duration(input_val)
            print(f"  parse_duration({repr(input_val):15}) -> {result:8.1f}s  ✗ (should have failed)")
        except ValueError as e:
            print(f"  parse_duration({repr(input_val):15}) -> ValueError ✓")


def test_duration_formatting():
    """Test the duration formatting function."""
    print("\n=== Testing Duration Formatting ===\n")
    
    test_cases = [
        (0, "0s", "0 seconds"),
        (30, "30s", "30 seconds"),
        (60, "1m", "1 minute"),
        (90, "1m30s", "1 minute 30 seconds"),
        (3600, "1h", "1 hour"),
        (3661, "1h1m1s", "1 hour 1 minute 1 second"),
        (7320, "2h2m", "2 hours 2 minutes"),
    ]
    
    for seconds, expected_compact, expected_verbose in test_cases:
        compact = format_duration(seconds, compact=True)
        verbose = format_duration(seconds, compact=False)
        
        print(f"  {seconds:5}s -> compact: {compact:10} {'✓' if compact == expected_compact else f'✗ ({expected_compact})'}")
        print(f"         -> verbose: {verbose:30} {'✓' if verbose == expected_verbose else f'✗ ({expected_verbose})'}")


def test_simulation_tracker_export():
    """Test exporting a SimulationTracker node with new telemetry options."""
    print("\n=== Testing SimulationTracker Export ===\n")
    
    from export_system.node_exporters.simulation_tracker_exporter import SimulationTrackerExporter
    
    # Test different telemetry configurations
    test_configs = [
        {
            "name": "Time-based with stats",
            "widgets_values": [1000, 0.95, "time", "10s", True],
            "expected": {
                "MAX_EPISODES": 1000,
                "SUCCESS_THRESHOLD": 0.95,
                "TELEMETRY_MODE": "'time'",
                "TELEMETRY_INTERVAL": "'10s'",
                "TELEMETRY_STATS": True,
            }
        },
        {
            "name": "Step-based without stats",
            "widgets_values": [500, 0.9, "steps", "100", False],
            "expected": {
                "MAX_EPISODES": 500,
                "SUCCESS_THRESHOLD": 0.9,
                "TELEMETRY_MODE": "'steps'",
                "TELEMETRY_INTERVAL": "'100'",
                "TELEMETRY_STATS": False,
            }
        },
        {
            "name": "Episode-based with complex time",
            "widgets_values": [2000, 0.99, "episodes", "5", True],
            "expected": {
                "MAX_EPISODES": 2000,
                "SUCCESS_THRESHOLD": 0.99,
                "TELEMETRY_MODE": "'episodes'",
                "TELEMETRY_INTERVAL": "'5'",
                "TELEMETRY_STATS": True,
            }
        },
        {
            "name": "Default values (empty widgets)",
            "widgets_values": [],
            "expected": {
                "MAX_EPISODES": 1000,
                "SUCCESS_THRESHOLD": 0.95,
                "TELEMETRY_MODE": "'time'",
                "TELEMETRY_INTERVAL": "'10s'",
                "TELEMETRY_STATS": True,
            }
        },
    ]
    
    for config in test_configs:
        print(f"  Testing: {config['name']}")
        node_data = {"widgets_values": config["widgets_values"]}
        
        template_vars = SimulationTrackerExporter.prepare_template_vars(
            "sim_tracker_1", node_data, {}, None, None, None
        )
        
        # Check all expected values
        all_match = True
        for key, expected_val in config["expected"].items():
            actual_val = template_vars.get(key)
            if actual_val != expected_val:
                print(f"    ✗ {key}: expected {expected_val}, got {actual_val}")
                all_match = False
        
        if all_match:
            print(f"    ✓ All template variables correct")
    
    # Check imports
    print("\n  Checking imports:")
    imports = SimulationTrackerExporter.get_imports()
    required_imports = [
        "import time",
        "import statistics", 
        "from framework.time_utils import parse_duration"
    ]
    
    for imp in required_imports:
        if imp in imports:
            print(f"    ✓ {imp}")
        else:
            print(f"    ✗ Missing: {imp}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SimulationTracker Telemetry System Test")
    print("="*60)
    
    test_time_parsing()
    test_duration_formatting()
    test_simulation_tracker_export()
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()