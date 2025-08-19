#!/usr/bin/env python3
"""
Test script to verify the robust timeout mechanism works correctly.
This simulates a deadlocked async loop to ensure the timeout still triggers.
"""

import asyncio
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export_system.templates.framework.graph_runner import GraphRunner
from export_system.templates.framework.base_nodes import QueueNode
from export_system.templates.framework.exceptions import CauseExitException


class DeadlockNode(QueueNode):
    """A node that creates a deadlock by blocking the async loop"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=[])
        self.setup_outputs([])
    
    async def run(self):
        """Simulate a deadlock by doing a blocking sleep"""
        print(f"Node {self.node_id}: Starting infinite blocking loop...")
        # This will block the async loop completely
        while True:
            # Synchronous sleep that blocks the event loop
            time.sleep(1)
            print(f"Node {self.node_id}: Still blocking...")
    
    async def compute(self, **kwargs):
        return {}


class NormalNode(QueueNode):
    """A normal node that works correctly"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=[])
        self.setup_outputs([])
    
    async def run(self):
        """Normal async operation"""
        print(f"Node {self.node_id}: Starting normal operation...")
        while self.running:
            await asyncio.sleep(0.5)
            print(f"Node {self.node_id}: Working normally...")
    
    async def compute(self, **kwargs):
        return {}


async def test_timeout_with_deadlock():
    """Test that timeout works even when a node deadlocks the async loop"""
    print("=" * 60)
    print("TEST 1: Timeout with deadlocked node")
    print("=" * 60)
    
    runner = GraphRunner()
    
    # Add a deadlock node and a normal node
    deadlock_node = DeadlockNode("deadlock_1")
    normal_node = NormalNode("normal_1")
    
    runner.add_node(deadlock_node)
    runner.add_node(normal_node)
    
    print("Starting graph with 3-second timeout...")
    print("The deadlock node will block the async loop, but timeout should still work.")
    
    start_time = time.time()
    try:
        await runner.run(duration=3.0)
    except Exception as e:
        print(f"Exception: {e}")
    
    elapsed = time.time() - start_time
    print(f"Graph stopped after {elapsed:.1f} seconds")
    print(f"Exit reason: {runner.exit_reason}")
    
    if runner.exit_reason == "timeout" and elapsed < 4.0:
        print("✅ TEST PASSED: Timeout worked despite deadlock!")
    else:
        print("❌ TEST FAILED: Timeout did not work properly")
    
    print()


async def test_normal_timeout():
    """Test that timeout works with normal async nodes"""
    print("=" * 60)
    print("TEST 2: Normal timeout with async nodes")
    print("=" * 60)
    
    runner = GraphRunner()
    
    # Add only normal nodes
    normal_node1 = NormalNode("normal_1")
    normal_node2 = NormalNode("normal_2")
    
    runner.add_node(normal_node1)
    runner.add_node(normal_node2)
    
    print("Starting graph with 2-second timeout...")
    
    start_time = time.time()
    try:
        await runner.run(duration=2.0)
    except Exception as e:
        print(f"Exception: {e}")
    
    elapsed = time.time() - start_time
    print(f"Graph stopped after {elapsed:.1f} seconds")
    print(f"Exit reason: {runner.exit_reason}")
    
    if runner.exit_reason == "timeout" and 1.9 < elapsed < 2.5:
        print("✅ TEST PASSED: Normal timeout worked!")
    else:
        print("❌ TEST FAILED: Timeout timing was off")
    
    print()


async def main():
    """Run all timeout tests"""
    print("\n" + "=" * 60)
    print("ROBUST TIMEOUT MECHANISM TEST")
    print("=" * 60 + "\n")
    
    # Test normal timeout
    await test_normal_timeout()
    
    # Test timeout with deadlock
    # NOTE: This test will actually block the async loop, 
    # so it demonstrates the robustness of the timeout
    await test_timeout_with_deadlock()
    
    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())