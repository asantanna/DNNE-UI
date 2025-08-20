#!/usr/bin/env python3
"""
Test script for training telemetry in EpochTracker node
"""

import sys
import os
import subprocess
import time

def main():
    """Test training telemetry with MNIST workflow"""
    
    # Change to export directory
    export_dir = "/home/asantanna/DNNE/DNNE-UI/export_system/exports/MNIST_Test"
    if not os.path.exists(export_dir):
        print(f"❌ Export directory not found: {export_dir}")
        print("Please run: python claude_scripts/programmatic_export.py MNIST_Test")
        return 1
    
    os.chdir(export_dir)
    print(f"📂 Changed to: {export_dir}")
    
    # Run with telemetry enabled for 2 epochs
    cmd = [
        "python", "runner.py",
        "--epochs", "2",
        "--enable-telemetry", "67",  # EpochTracker node ID
        "--override", "67:telemetry_batch_window=50",  # Report every 50 batches
        "--timeout", "30s"  # Timeout after 30 seconds
    ]
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Training completed successfully!")
        else:
            print(f"\n⚠️ Training exited with code: {result.returncode}")
            
        # Check for telemetry files
        print("\n📊 Checking for telemetry files...")
        telemetry_base = "/home/asantanna/DNNE/DNNE-UI/remote_clients"
        
        # Look for recent telemetry files
        telemetry_cmd = f"find {telemetry_base} -name 'node_67.dat' -mmin -5 2>/dev/null | head -1"
        telemetry_file = subprocess.run(telemetry_cmd, shell=True, capture_output=True, text=True).stdout.strip()
        
        if telemetry_file:
            print(f"✅ Found telemetry file: {telemetry_file}")
            
            # Show last few lines
            print("\n📈 Last 10 telemetry entries:")
            subprocess.run(f"tail -10 {telemetry_file}", shell=True)
            
            # Count different metric types
            print("\n📊 Telemetry metrics summary:")
            subprocess.run(f"cut -d'|' -f2 {telemetry_file} | sort | uniq -c | sort -rn | head -20", shell=True)
        else:
            print("⚠️ No telemetry files found. Make sure:")
            print("  1. Agent client is running: python dnne_agent/dnne_agent_client.py")
            print("  2. DNNE server is running")
            
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())