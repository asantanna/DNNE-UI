#!/usr/bin/env python3
"""
Test script to verify the MultiWaiter efficiency improvements.
Tests both OR and Concat nodes with the new pattern.
"""

import asyncio
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add export templates to path for framework imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'export_system', 'templates'))

from export_system.templates.framework.base_nodes import QueueNode
from export_system.templates.framework.multi_waiter import MultiWaiter
from export_system.templates.framework.graph_runner import GraphRunner
from typing import Dict, Any


class TestProducerNode(QueueNode):
    """Produces data at regular intervals"""
    
    def __init__(self, node_id: str, interval: float = 0.1):
        super().__init__(node_id)
        self.interval = interval
        self.setup_inputs(required=[])
        self.setup_outputs(["output"])
        self.counter = 0
    
    async def compute(self, **inputs) -> Dict[str, Any]:
        # This won't be called since we override run()
        return {}
    
    async def run(self):
        """Produce data continuously"""
        self.running = True
        print(f"Producer {self.node_id} starting (interval={self.interval}s)")
        
        try:
            while self.running:
                await asyncio.sleep(self.interval)
                self.counter += 1
                data = f"{self.node_id}_data_{self.counter}"
                await self.send_output("output", data)
                print(f"  Producer {self.node_id}: Sent {data}")
        except asyncio.CancelledError:
            print(f"Producer {self.node_id} cancelled")
            raise
        finally:
            self.running = False


class TestORNode(QueueNode):
    """OR node using new MultiWaiter pattern"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id, wait_mode="any")
        self.setup_inputs(required=[])
        self.setup_outputs(["output"])
        
        # Create input queues
        self.input_queues["input_a"] = asyncio.Queue(maxsize=2)
        self.input_queues["input_b"] = asyncio.Queue(maxsize=2)
        
        # Create MultiWaiter
        from export_system.templates.framework.multi_waiter import MultiWaiter
        self.input_waiter = MultiWaiter(
            ["input_a", "input_b"],
            self.input_queues,
            wait_mode="any"
        )
        self.receive_count = 0
    
    async def compute(self, **inputs) -> Dict[str, Any]:
        # Not used since we override run()
        return {}
    
    async def run(self):
        """Custom run using MultiWaiter"""
        self.running = True
        print(f"OR node {self.node_id} starting")
        
        try:
            while self.running:
                # Wait for ANY input
                data, source = await self.input_waiter.get()
                self.receive_count += 1
                print(f"  OR {self.node_id}: Received '{data}' from {source} (#{self.receive_count})")
                
                # Forward output
                await self.send_output("output", f"OR_{data}")
                
        except asyncio.CancelledError:
            print(f"OR node {self.node_id} cancelled")
            raise
        finally:
            self.running = False


class TestConcatNode(QueueNode):
    """Concat node using new MultiWaiter pattern"""
    
    def __init__(self, node_id: str, mode: str = "all"):
        wait_mode = "all" if mode == "wait for all" else "any"
        super().__init__(node_id, wait_mode=wait_mode)
        self.mode = mode
        self.setup_inputs(required=[])
        self.setup_outputs(["output"])
        
        # Create input queues
        self.input_queues["input_a"] = asyncio.Queue(maxsize=2)
        self.input_queues["input_b"] = asyncio.Queue(maxsize=2)
        
        # Create MultiWaiter
        from export_system.templates.framework.multi_waiter import MultiWaiter
        self.input_waiter = MultiWaiter(
            ["input_a", "input_b"],
            self.input_queues,
            wait_mode=wait_mode
        )
        self.receive_count = 0
    
    async def compute(self, **inputs) -> Dict[str, Any]:
        # Not used since we override run()
        return {}
    
    async def run(self):
        """Custom run using MultiWaiter"""
        self.running = True
        print(f"Concat node {self.node_id} starting (mode={self.mode})")
        
        try:
            while self.running:
                if self.wait_mode == "all":
                    # Wait for ALL inputs
                    inputs = await self.input_waiter.get()
                    self.receive_count += 1
                    print(f"  Concat {self.node_id}: Got all inputs: {inputs} (#{self.receive_count})")
                    
                    # Concatenate (simulated)
                    output = f"CONCAT_[{inputs['input_a']}+{inputs['input_b']}]"
                else:
                    # Wait for ANY input
                    data, source = await self.input_waiter.get()
                    self.receive_count += 1
                    print(f"  Concat {self.node_id}: Got '{data}' from {source} (#{self.receive_count})")
                    
                    # Output with padding (simulated)
                    output = f"CONCAT_PADDED_{data}"
                
                await self.send_output("output", output)
                
        except asyncio.CancelledError:
            print(f"Concat node {self.node_id} cancelled")
            raise
        finally:
            self.running = False


class TestConsumerNode(QueueNode):
    """Consumes and logs data"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["input"])
        self.setup_outputs([])
        self.consume_count = 0
    
    async def compute(self, **inputs) -> Dict[str, Any]:
        self.consume_count += 1
        data = inputs.get("input")
        print(f"    Consumer {self.node_id}: Consumed '{data}' (#{self.consume_count})")
        return {}


async def test_or_node():
    """Test OR node with multiple producers"""
    print("\n" + "="*60)
    print("TEST 1: OR Node with MultiWaiter")
    print("="*60)
    
    runner = GraphRunner()
    
    # Create nodes
    producer1 = TestProducerNode("prod1", interval=0.2)
    producer2 = TestProducerNode("prod2", interval=0.3)
    or_node = TestORNode("or1")
    consumer = TestConsumerNode("cons1")
    
    # Add nodes
    runner.add_node(producer1)
    runner.add_node(producer2)
    runner.add_node(or_node)
    runner.add_node(consumer)
    
    # Wire connections
    runner.wire_nodes([
        ("prod1", "output", "or1", "input_a"),
        ("prod2", "output", "or1", "input_b"),
        ("or1", "output", "cons1", "input")
    ])
    
    print("\nRunning for 0.5 seconds...")
    print("Expected: OR node efficiently routes inputs without task creation/destruction")
    
    # Run for 0.5 seconds (quick unit test)
    await runner.run(duration=0.5)
    
    print(f"\nResults: OR received {or_node.receive_count} inputs, Consumer processed {consumer.consume_count}")
    print("✅ OR node test complete\n")


async def test_concat_all():
    """Test Concat node in 'wait for all' mode"""
    print("\n" + "="*60)
    print("TEST 2: Concat Node with 'wait for all' mode")
    print("="*60)
    
    runner = GraphRunner()
    
    # Create nodes
    producer1 = TestProducerNode("prod1", interval=0.2)
    producer2 = TestProducerNode("prod2", interval=0.3)
    concat_node = TestConcatNode("concat1", mode="wait for all")
    consumer = TestConsumerNode("cons1")
    
    # Add nodes
    runner.add_node(producer1)
    runner.add_node(producer2)
    runner.add_node(concat_node)
    runner.add_node(consumer)
    
    # Wire connections
    runner.wire_nodes([
        ("prod1", "output", "concat1", "input_a"),
        ("prod2", "output", "concat1", "input_b"),
        ("concat1", "output", "cons1", "input")
    ])
    
    print("\nRunning for 0.5 seconds...")
    print("Expected: Concat waits for both inputs before outputting")
    
    # Run for 0.5 seconds (quick unit test)
    await runner.run(duration=0.5)
    
    print(f"\nResults: Concat received {concat_node.receive_count} complete sets, Consumer processed {consumer.consume_count}")
    print("✅ Concat 'wait for all' test complete\n")


async def test_concat_any():
    """Test Concat node in 'as available' mode"""
    print("\n" + "="*60)
    print("TEST 3: Concat Node with 'as available' mode")
    print("="*60)
    
    runner = GraphRunner()
    
    # Create nodes
    producer1 = TestProducerNode("prod1", interval=0.2)
    producer2 = TestProducerNode("prod2", interval=0.5)  # Slower
    concat_node = TestConcatNode("concat1", mode="as available")
    consumer = TestConsumerNode("cons1")
    
    # Add nodes
    runner.add_node(producer1)
    runner.add_node(producer2)
    runner.add_node(concat_node)
    runner.add_node(consumer)
    
    # Wire connections
    runner.wire_nodes([
        ("prod1", "output", "concat1", "input_a"),
        ("prod2", "output", "concat1", "input_b"),
        ("concat1", "output", "cons1", "input")
    ])
    
    print("\nRunning for 0.5 seconds...")
    print("Expected: Concat outputs immediately when any input arrives")
    
    # Run for 0.5 seconds (quick unit test)
    await runner.run(duration=0.5)
    
    print(f"\nResults: Concat received {concat_node.receive_count} inputs, Consumer processed {consumer.consume_count}")
    print("✅ Concat 'as available' test complete\n")


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MULTIWAITER EFFICIENCY TEST SUITE")
    print("="*60)
    
    await test_or_node()
    await test_concat_all()
    await test_concat_any()
    
    print("="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
    print("\nKey improvements:")
    print("- No constant task creation/destruction")
    print("- Persistent listener tasks for 'any' mode")
    print("- Simple sequential waits for 'all' mode")
    print("- Uniform API across all node types")


if __name__ == "__main__":
    asyncio.run(main())