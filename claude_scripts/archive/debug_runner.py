#!/usr/bin/env python3
"""
Debug version of MNIST runner with enhanced logging
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path("../export_system/exports/MNIST-Test")))

import asyncio
import logging
import time

# Configure more verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from framework.base import GraphRunner
from nodes import *

class DebugGraphRunner(GraphRunner):
    """Enhanced GraphRunner with debug logging"""
    
    def __init__(self):
        super().__init__()
        self.last_stats_time = time.time()
        
    async def run(self, duration=None):
        """Run with periodic stats reporting"""
        self.logger.info("Starting DEBUG graph execution")
        
        # Start all nodes
        for node in self.nodes.values():
            task = asyncio.create_task(node.run())
            self.tasks.append(task)
        
        # Start stats reporting task
        stats_task = asyncio.create_task(self._periodic_stats())
        self.tasks.append(stats_task)
        
        try:
            if duration:
                await asyncio.sleep(duration)
                self.logger.info(f"Stopping after {duration}s")
            else:
                # Run until cancelled
                await asyncio.gather(*self.tasks)
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            # Cancel all tasks
            for task in self.tasks:
                task.cancel()
            
            # Wait for cancellation
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.logger.info("All nodes stopped")
    
    async def _periodic_stats(self):
        """Report stats every 30 seconds"""
        try:
            while True:
                await asyncio.sleep(30)
                self.logger.info("=== PERIODIC STATS ===")
                stats = self.get_stats()
                for node_id, node_stats in stats.items():
                    self.logger.info(f"  {node_id}: {node_stats['compute_count']} computations, "
                                   f"avg time: {node_stats['last_compute_time']:.3f}s, "
                                   f"running: {node_stats['running']}")
                
                # Check queue sizes
                for node_id, node in self.nodes.items():
                    queue_sizes = {}
                    for input_name, queue in node.input_queues.items():
                        queue_sizes[input_name] = queue.qsize()
                    if queue_sizes:
                        self.logger.info(f"  {node_id} queue sizes: {queue_sizes}")
                
                self.logger.info("=== END STATS ===")
        except asyncio.CancelledError:
            pass

async def main():
    """Main execution function with debug logging"""
    print("🚀 Starting DNNE DEBUG Execution")
    print("=" * 60)

    # Create nodes
    node_37 = MNISTDatasetNode_37("37")
    node_45 = TrainingStepNode_45("45")
    node_38 = BatchSamplerNode_38("38")
    node_40 = NetworkNode_40("40")
    node_50 = GetBatchNode_50("50")
    node_51 = LossNode_51("51")
    node_55 = EpochTrackerNode_55("55")
    node_44 = SGDOptimizerNode_44("44")

    # Create debug runner
    runner = DebugGraphRunner()

    # Add nodes to runner
    runner.add_node(node_37)
    runner.add_node(node_45)
    runner.add_node(node_38)
    runner.add_node(node_40)
    runner.add_node(node_50)
    runner.add_node(node_51)
    runner.add_node(node_55)
    runner.add_node(node_44)

    # Wire connections
    connections = [
        ("37", "dataset", "38", "dataset"),
        ("37", "schema", "38", "schema"),
        ("40", "model", "44", "network"),
        ("44", "optimizer", "45", "optimizer"),
        ("38", "dataloader", "50", "dataloader"),
        ("38", "schema", "50", "schema"),
        ("50", "images", "40", "input"),
        ("40", "output", "51", "predictions"),
        ("51", "loss", "45", "loss"),
        ("50", "labels", "51", "labels"),
        ("50", "epoch_stats", "55", "epoch_stats"),
        ("51", "loss", "55", "loss"),
        ("51", "accuracy", "55", "accuracy"),
    ]
    runner.wire_nodes(connections)

    # Run the graph for 60 seconds to debug
    try:
        await runner.run(duration=60.0)
    except KeyboardInterrupt:
        print('\n🛑 Stopped by user')

    # Show final statistics
    print('\n📊 Final Statistics:')
    stats = runner.get_stats()
    for node_id, node_stats in stats.items():
        print(f'  {node_id}: {node_stats["compute_count"]} computations, '
              f'avg time: {node_stats["last_compute_time"]:.3f}s, '
              f'running: {node_stats["running"]}')

if __name__ == '__main__':
    asyncio.run(main())