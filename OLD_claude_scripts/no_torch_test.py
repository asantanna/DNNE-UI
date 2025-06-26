#!/usr/bin/env python3
"""
Test queue system without PyTorch dependencies
"""
print("Starting no-torch test...")

import asyncio
import time
import random
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from asyncio import Queue
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')

# Include the queue framework inline
class QueueNode(ABC):
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.input_queues: Dict[str, Queue] = {}
        self.output_subscribers: Dict[str, List[Queue]] = {}
        self.required_inputs: List[str] = []
        self.running = False
        self.compute_count = 0
        self.logger = logging.getLogger(f"Node.{node_id}")
    
    def setup_inputs(self, required: List[str]):
        self.required_inputs = required
        for input_name in required:
            self.input_queues[input_name] = Queue(maxsize=100)
    
    def setup_outputs(self, outputs: List[str]):
        for output_name in outputs:
            self.output_subscribers[output_name] = []
    
    async def send_output(self, output_name: str, value: Any):
        if output_name in self.output_subscribers:
            for queue in self.output_subscribers[output_name]:
                await queue.put(value)
    
    @abstractmethod
    async def compute(self, **inputs) -> Dict[str, Any]:
        pass
    
    async def run(self):
        self.running = True
        self.logger.info(f"Starting node")
        try:
            while self.running:
                inputs = {}
                for input_name in self.required_inputs:
                    value = await self.input_queues[input_name].get()
                    inputs[input_name] = value
                
                outputs = await self.compute(**inputs)
                self.compute_count += 1
                
                for output_name, value in outputs.items():
                    await self.send_output(output_name, value)
        except asyncio.CancelledError:
            self.logger.info(f"Cancelled")
            raise

class SensorNode(QueueNode):
    def __init__(self, node_id: str, update_rate: float):
        super().__init__(node_id)
        self.update_rate = update_rate
        self.update_interval = 1.0 / update_rate
    
    async def run(self):
        self.running = True
        self.logger.info(f"Starting sensor at {self.update_rate}Hz")
        try:
            while self.running:
                start_time = time.time()
                outputs = await self.compute()
                self.compute_count += 1
                
                for output_name, value in outputs.items():
                    await self.send_output(output_name, value)
                
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            self.logger.info(f"Cancelled")
            raise

# Test nodes using plain Python lists instead of tensors
class FakeDataSensor(SensorNode):
    """Generates fake data without PyTorch"""
    def __init__(self, node_id: str):
        super().__init__(node_id, update_rate=2.0)
        self.setup_outputs(["data", "label"])
        self.batch_size = 32
        
    async def compute(self) -> Dict[str, Any]:
        # Simulate batch data with plain Python lists
        data = [[random.random() for _ in range(784)] for _ in range(self.batch_size)]
        labels = [random.randint(0, 9) for _ in range(self.batch_size)]
        return {
            "data": data,
            "label": labels
        }

class FakeProcessor(QueueNode):
    """Process without PyTorch"""
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["data"])
        self.setup_outputs(["stats"])
        
    async def compute(self, data) -> Dict[str, Any]:
        # Simple statistics
        flat = [val for batch in data for val in batch]
        mean = sum(flat) / len(flat)
        return {"stats": {"mean": mean, "count": len(data)}}

class StatsDisplay(QueueNode):
    """Display statistics"""
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["stats"])
        self.setup_outputs([])
        
    async def compute(self, stats) -> Dict[str, Any]:
        self.logger.info(f"[{self.compute_count}] Stats: mean={stats['mean']:.4f}, batches={stats['count']}")
        return {}

async def main():
    print("\n🚀 Queue System Test (No PyTorch)")
    print("=" * 60)
    
    # Create nodes
    sensor = FakeDataSensor("sensor")
    processor = FakeProcessor("processor")
    display = StatsDisplay("display")
    
    # Wire connections
    sensor.output_subscribers["data"].append(processor.input_queues["data"])
    processor.output_subscribers["stats"].append(display.input_queues["stats"])
    
    # Run
    tasks = [
        asyncio.create_task(sensor.run()),
        asyncio.create_task(processor.run()),
        asyncio.create_task(display.run())
    ]
    
    try:
        await asyncio.sleep(5)
        print("\n✅ Success! Queue system works without PyTorch")
    except KeyboardInterrupt:
        print("\n🛑 Stopped")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    
    print(f"\nFinal counts:")
    print(f"  Sensor: {sensor.compute_count}")
    print(f"  Processor: {processor.compute_count}")
    print(f"  Display: {display.compute_count}")

if __name__ == "__main__":
    print("This test runs without PyTorch to verify the queue system")
    asyncio.run(main())
