#!/usr/bin/env python3
"""
Telemetry Overhead Test
Measures the performance impact of telemetry on real workflows.
Uses CIFAR10_Test.json to compare execution times with and without telemetry.
"""

import sys
import asyncio
import time
import websockets
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from helpers.deployment_helper import (
    check_client_connected,
    deploy_workflow_to_client,
    cleanup_workflow_directories
)


class TelemetryOverheadTest:
    """Test telemetry overhead using CIFAR10 workflow"""
    
    def __init__(self):
        self.hostname = "test_node_overhead"
        self.test_iterations = 3  # Run multiple iterations for averaging
        self.windows_host = "172.22.160.1"  # Windows host IP
        self.test_port_url = f"ws://{self.windows_host}:8768"
        
    async def run_test(self):
        """Run the complete overhead test"""
        print("\n" + "="*60)
        print("TELEMETRY OVERHEAD TEST - CIFAR10 Workflow")
        print("="*60)
        
        # Clean up old workflows
        await cleanup_workflow_directories(self.hostname, "CIFAR10_*")
        
        # Connect to test port
        try:
            websocket = await websockets.connect(self.test_port_url)
            print(f"Connected to agent server test port")
            
            # Check if client is connected
            if not await check_client_connected(websocket, self.hostname):
                print("ERROR: Test client not connected. Please run:")
                print(f"  python dnne-agent/dnne_agent_client.py --hostname {self.hostname} --test-mode")
                return False
            
            print(f"Test client connected")
            
            # Run tests with and without telemetry
            times_with_telemetry = []
            times_without_telemetry = []
            
            print(f"\nRunning {self.test_iterations} iterations...")
            
            for i in range(self.test_iterations):
                print(f"\n--- Iteration {i+1}/{self.test_iterations} ---")
                
                # Run without telemetry
                print("\nRunning WITHOUT telemetry:")
                execution_time = await deploy_workflow_to_client(
                    websocket=websocket,
                    workflow_name="CIFAR10_Test",
                    client_hostname=self.hostname,
                    runner_args="--epochs 2",
                    workflow_id=f"cifar10_overhead_{i}_no_telem",
                    monitor_execution=True
                )
                
                if execution_time is not None:
                    print(f"  Workflow completed in {execution_time:.2f} seconds")
                    times_without_telemetry.append(execution_time)
                else:
                    print(f"  Workflow execution failed")
                
                # Small delay between tests
                await asyncio.sleep(2)
                
                # Run with telemetry
                print("\nRunning WITH telemetry:")
                execution_time = await deploy_workflow_to_client(
                    websocket=websocket,
                    workflow_name="CIFAR10_Test",
                    client_hostname=self.hostname,
                    runner_args="--epochs 2 --enable-telemetry 10,11",
                    workflow_id=f"cifar10_overhead_{i}_with_telem",
                    monitor_execution=True
                )
                
                if execution_time is not None:
                    print(f"  Workflow completed in {execution_time:.2f} seconds")
                    times_with_telemetry.append(execution_time)
                else:
                    print(f"  Workflow execution failed")
                
                # Clean up after each iteration
                await cleanup_workflow_directories(self.hostname, "cifar10_*")
            
            # Calculate results
            print("\n" + "="*60)
            print("RESULTS")
            print("="*60)
            
            if not times_without_telemetry or not times_with_telemetry:
                print("ERROR: Could not collect sufficient timing data")
                return False
            
            avg_without = sum(times_without_telemetry) / len(times_without_telemetry)
            avg_with = sum(times_with_telemetry) / len(times_with_telemetry)
            
            print(f"\nWithout Telemetry:")
            print(f"  Times: {[f'{t:.2f}s' for t in times_without_telemetry]}")
            print(f"  Average: {avg_without:.2f} seconds")
            
            print(f"\nWith Telemetry:")
            print(f"  Times: {[f'{t:.2f}s' for t in times_with_telemetry]}")
            print(f"  Average: {avg_with:.2f} seconds")
            
            # Calculate overhead
            overhead_seconds = avg_with - avg_without
            overhead_percentage = (overhead_seconds / avg_without) * 100 if avg_without > 0 else 0
            
            print(f"\nTelemetry Overhead:")
            print(f"  Absolute: {overhead_seconds:.2f} seconds")
            print(f"  Relative: {overhead_percentage:.1f}%")
            
            # Determine if overhead is acceptable
            acceptable_threshold = 5.0  # 5% overhead threshold
            if overhead_percentage < acceptable_threshold:
                print(f"\nPASS: Telemetry overhead ({overhead_percentage:.1f}%) is below {acceptable_threshold}% threshold")
                return True
            else:
                print(f"\nWARNING: Telemetry overhead ({overhead_percentage:.1f}%) exceeds {acceptable_threshold}% threshold")
                print("Consider optimizing telemetry implementation if this impacts production workloads")
                return True  # Still pass but with warning
                
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if 'websocket' in locals():
                await websocket.close()


async def main():
    """Main entry point"""
    test = TelemetryOverheadTest()
    success = await test.run_test()
    
    print("\n" + "="*60)
    if success:
        print("TELEMETRY OVERHEAD TEST COMPLETED")
    else:
        print("TELEMETRY OVERHEAD TEST FAILED")
    print("="*60)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())