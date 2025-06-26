#!/usr/bin/env python3
"""
Test export system with robotics nodes (no ML dependencies)
"""

from export_system.graph_exporter import GraphExporter
from export_system.node_exporters import register_all_exporters
import time

def test_robotics_simple():
    print("Testing Queue Export with Simple Robotics Workflow")
    print("=" * 60)
    
    # Create exporter
    exporter = GraphExporter()
    register_all_exporters(exporter)
    
    # Simple robotics workflow - camera to display
    workflow = {
        "nodes": [
            {
                "id": "cam1",
                "class_type": "CameraSensor",
                "inputs": {
                    "fps": 5.0,  # 5 FPS
                    "resolution": "320x240",
                    "use_real_camera": False
                }
            },
            {
                "id": "display1",
                "class_type": "Display", 
                "inputs": {
                    "display_type": "value",
                    "log_interval": 2  # Show every 2nd frame
                }
            }
        ],
        "links": [
            [1, "cam1", 0, "display1", 0]  # Camera image -> Display
        ],
        "metadata": {
            "name": "Simple Camera Test",
            "description": "Camera sensor to display without ML",
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Export
    script = exporter.export_workflow(workflow)
    
    # Save
    output_file = "test_camera_queue.py"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(script)
    
    print(f"✅ Exported to: {output_file}")
    print(f"   Generated {len(script.splitlines())} lines")
    
    # Also test a multi-sensor workflow
    print("\n" + "=" * 60)
    print("Testing Multi-Sensor Workflow")
    
    multi_workflow = {
        "nodes": [
            {
                "id": "cam",
                "class_type": "CameraSensor",
                "inputs": {"fps": 10.0, "resolution": "640x480"}
            },
            {
                "id": "imu",
                "class_type": "IMUSensor",
                "inputs": {"sample_rate": 50.0}
            },
            {
                "id": "display_cam",
                "class_type": "Display",
                "inputs": {"display_type": "value", "log_interval": 5}
            },
            {
                "id": "display_imu",
                "class_type": "Display",
                "inputs": {"display_type": "value", "log_interval": 10}
            }
        ],
        "links": [
            [1, "cam", 0, "display_cam", 0],    # Camera -> Display
            [2, "imu", 0, "display_imu", 0]     # IMU accel -> Display
        ],
        "metadata": {
            "name": "Multi-Sensor Test",
            "description": "Parallel sensors at different rates"
        }
    }
    
    script2 = exporter.export_workflow(multi_workflow)
    
    output_file2 = "test_multisensor_queue.py"
    with open(output_file2, "w", encoding="utf-8") as f:
        f.write(script2)
    
    print(f"✅ Exported to: {output_file2}")
    print(f"   Generated {len(script2.splitlines())} lines")
    
    print("\n✅ Export tests complete!")
    print("\nYou can now run:")
    print(f"  python {output_file}")
    print(f"  python {output_file2}")
    print("\nThese should run immediately without PyTorch dependencies.")

if __name__ == "__main__":
    test_robotics_simple()
