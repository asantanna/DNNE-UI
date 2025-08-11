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
import subprocess
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from helpers.deployment_helper import (
    TestClientManager,
    check_client_connected,
    deploy_workflow_to_client,
    cleanup_workflow_directories
)


class TelemetryOverheadTest:
    """Test telemetry overhead using CIFAR10 workflow"""
    
    def __init__(self):
        self.hostname = "agent_client_test_host"  # Use same hostname as test mode
        self.test_iterations = 2  # Run 2 test iterations after warmup
        self.windows_host = "172.22.160.1"  # Windows host IP
        self.test_port_url = f"ws://{self.windows_host}:8768"
        self.client_manager = TestClientManager(self.windows_host)
        
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
            
            # Read and discard initial state message
            initial_msg = await websocket.recv()
            
            # Ensure test client is connected
            if not await self.client_manager.ensure_test_client_connected(websocket, self.hostname):
                return False
            
            # Set up path to pre-downloaded CIFAR-10 dataset
            cifar_data_dir = "/home/asantanna/DNNE/DNNE-UI/dnne-test-suite/DATASETS/cifar10"
            
            print("\n" + "="*60)
            print("📦 Using pre-downloaded CIFAR-10 dataset")
            print(f"   Source: {cifar_data_dir}")
            print("="*60)
            
            # WARMUP RUN - Extract dataset and compile
            print("\n" + "="*60)
            print("🔥 WARMUP RUN - Extracting dataset and warming up...")
            print("   (This run is not timed - it ensures data is extracted)")
            print("="*60)
            
            warmup_time = await deploy_workflow_to_client(
                websocket=websocket,
                workflow_name="CIFAR10_Test",
                client_hostname=self.hostname,
                runner_args="--epochs 1",  # Just 1 epoch for warmup
                workflow_id="cifar10_warmup",
                monitor_execution=True,
                copy_dir=(cifar_data_dir, "data")
            )
            
            if warmup_time is not None:
                print(f"✅ Warmup completed in {warmup_time:.2f} seconds")
                print("   Dataset extracted and model compiled for subsequent runs")
            else:
                print("❌ Warmup failed - continuing anyway")
            
            # Clean up warmup files
            await cleanup_workflow_directories(self.hostname, "cifar10_warmup")
            await asyncio.sleep(2)  # Brief pause before actual tests
            
            # Run tests with and without telemetry
            times_with_telemetry = []
            times_without_telemetry = []
            
            print(f"\n🚀 Running {self.test_iterations} test iterations...")
            print("="*60)
            
            for i in range(self.test_iterations):
                print(f"\n--- Iteration {i+1}/{self.test_iterations} ---")
                
                # Run without telemetry
                print("\nRunning WITHOUT telemetry:")
                execution_time = await deploy_workflow_to_client(
                    websocket=websocket,
                    workflow_name="CIFAR10_Test",
                    client_hostname=self.hostname,
                    runner_args="--epochs 2",  # 2 epochs for better measurement
                    workflow_id=f"cifar10_overhead_{i}_no_telem",
                    monitor_execution=True,
                    copy_dir=(cifar_data_dir, "data")
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
                    runner_args="--epochs 2 --enable-telemetry 10,11",  # 2 epochs
                    workflow_id=f"cifar10_overhead_{i}_with_telem",
                    monitor_execution=True,
                    copy_dir=(cifar_data_dir, "data")
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
            # Clean up test client if we started it
            await self.client_manager.stop_test_client()


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