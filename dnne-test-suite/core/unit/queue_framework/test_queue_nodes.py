"""
Unit tests for DNNE queue framework.

Tests QueueNode base classes, async communication patterns, trigger-based
coordination, and the core async execution framework.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

from fixtures.test_utils import MockQueueNode, run_async_test, AsyncContextManager


class TestQueueNodeBase:
    """Test basic QueueNode functionality."""
    
    @pytest.mark.queue_framework
    def test_queue_node_initialization(self):
        """Test QueueNode initialization."""
        node = MockQueueNode("test_node_1")
        
        assert node.node_id == "test_node_1"
        assert isinstance(node.input_queues, dict)
        assert isinstance(node.output_queues, dict)
        assert len(node.input_queues) == 0
        assert len(node.output_queues) == 0
    
    @pytest.mark.queue_framework
    def test_input_setup(self):
        """Test input queue setup."""
        node = MockQueueNode("test_node")
        
        # Setup inputs
        node.setup_inputs(required=["input1", "input2"], optional=["input3"])
        
        # Should have created input queues
        assert "input1" in node.input_queues
        assert "input2" in node.input_queues
        assert "input3" in node.input_queues
        
        # Queues should be asyncio.Queue instances
        assert isinstance(node.input_queues["input1"], asyncio.Queue)
        assert isinstance(node.input_queues["input2"], asyncio.Queue)
        assert isinstance(node.input_queues["input3"], asyncio.Queue)
    
    @pytest.mark.queue_framework
    def test_output_setup(self):
        """Test output queue setup."""
        node = MockQueueNode("test_node")
        
        # Setup outputs
        node.setup_outputs(["output1", "output2", "ready_signal"])
        
        # Should have created output queues
        assert "output1" in node.output_queues
        assert "output2" in node.output_queues
        assert "ready_signal" in node.output_queues
        
        # Queues should be asyncio.Queue instances
        for queue in node.output_queues.values():
            assert isinstance(queue, asyncio.Queue)
    
    @pytest.mark.queue_framework
    @pytest.mark.asyncio
    async def test_basic_compute(self):
        """Test basic compute method functionality."""
        node = MockQueueNode("test_node")
        node.setup_outputs(["output1", "output2"])
        
        # Call compute method
        result = await node.compute(input_data="test_input")
        
        # Should return expected result
        assert isinstance(result, dict)
        assert "output1" in result
        assert "output2" in result
        
        # Mock returns mock_result_{output_name}
        assert result["output1"] == "mock_result_output1"
        assert result["output2"] == "mock_result_output2"


class TestAsyncCommunication:
    """Test async communication between queue nodes."""
    
    @pytest.mark.queue_framework
    @pytest.mark.asyncio
    async def test_queue_put_get(self):
        """Test basic queue put/get operations."""
        node = MockQueueNode("test_node")
        node.setup_inputs(required=["input1"])
        node.setup_outputs(["output1"])
        
        # Put data into input queue
        test_data = {"key": "value", "number": 42}
        await node.input_queues["input1"].put(test_data)
        
        # Get data from input queue
        received_data = await node.input_queues["input1"].get()
        
        # Data should be identical
        assert received_data == test_data
        assert received_data["key"] == "value"
        assert received_data["number"] == 42
    
    @pytest.mark.queue_framework
    @pytest.mark.asyncio
    async def test_queue_communication_between_nodes(self):
        """Test communication between two queue nodes."""
        # Create producer and consumer nodes
        producer = MockQueueNode("producer")
        consumer = MockQueueNode("consumer")
        
        producer.setup_outputs(["data"])
        consumer.setup_inputs(required=["data"])
        
        # Connect producer output to consumer input (simulation)
        shared_queue = asyncio.Queue()
        producer.output_queues["data"] = shared_queue
        consumer.input_queues["data"] = shared_queue
        
        # Producer sends data
        test_data = {"message": "hello", "value": 123}
        await producer.output_queues["data"].put(test_data)
        
        # Consumer receives data
        received_data = await consumer.input_queues["data"].get()
        
        assert received_data == test_data
        assert received_data["message"] == "hello"
        assert received_data["value"] == 123
    
    @pytest.mark.queue_framework
    @pytest.mark.asyncio
    async def test_queue_size_limits(self):
        """Test queue size limitations and backpressure."""
        node = MockQueueNode("test_node")
        node.setup_outputs(["output"])
        
        # Create small queue for testing
        small_queue = asyncio.Queue(maxsize=2)
        node.output_queues["output"] = small_queue
        
        # Fill queue to capacity
        await small_queue.put("item1")
        await small_queue.put("item2")
        
        # Queue should be full
        assert small_queue.full()
        
        # Next put should block (test with timeout)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(small_queue.put("item3"), timeout=0.1)
    
    @pytest.mark.queue_framework
    @pytest.mark.asyncio
    async def test_queue_fifo_ordering(self):
        """Test FIFO ordering of queue operations."""
        node = MockQueueNode("test_node")
        node.setup_inputs(required=["input"])
        
        queue = node.input_queues["input"]
        
        # Put items in order
        items = ["first", "second", "third", "fourth"]
        for item in items:
            await queue.put(item)
        
        # Get items - should be in same order
        received_items = []
        for _ in range(len(items)):
            received_items.append(await queue.get())
        
        assert received_items == items


class TestTriggerCoordination:
    """Test trigger-based coordination between nodes."""
    
    @pytest.mark.queue_framework
    @pytest.mark.asyncio
    async def test_trigger_signal_format(self):
        """Test trigger signal format and structure."""
        # Common trigger signal patterns
        trigger_signals = [
            {"signal_type": "ready", "timestamp": time.time()},
            {"signal_type": "ready", "source": "training_step", "metadata": {"loss": 1.5}},
            {"ready": True, "step": 42},
            "ready"  # Simple string signal
        ]
        
        for trigger in trigger_signals:
            # Trigger should be serializable and meaningful
            assert trigger is not None
            
            if isinstance(trigger, dict):
                # Dict triggers should have meaningful keys
                keys = list(trigger.keys())
                assert len(keys) > 0
                
                # Should have signal type or ready indicator
                has_signal_info = any(key in keys for key in 
                                    ["signal_type", "ready", "trigger", "step"])
                assert has_signal_info
    
    @pytest.mark.queue_framework
    @pytest.mark.asyncio
    async def test_trigger_based_execution(self):
        """Test trigger-based execution pattern."""
        # Simulate training step -> get batch coordination
        training_step = MockQueueNode("training_step")
        get_batch = MockQueueNode("get_batch")
        
        training_step.setup_outputs(["ready_signal"])
        get_batch.setup_inputs(required=["trigger"])
        
        # Connect trigger signal
        trigger_queue = asyncio.Queue()
        training_step.output_queues["ready_signal"] = trigger_queue
        get_batch.input_queues["trigger"] = trigger_queue
        
        # Training step sends ready signal
        ready_signal = {"signal_type": "ready", "timestamp": time.time()}
        await training_step.output_queues["ready_signal"].put(ready_signal)
        
        # Get batch receives trigger
        received_trigger = await get_batch.input_queues["trigger"].get()
        
        assert received_trigger == ready_signal
        assert received_trigger["signal_type"] == "ready"
    
    @pytest.mark.queue_framework
    @pytest.mark.asyncio
    async def test_multi_step_trigger_chain(self):
        """Test trigger chain across multiple nodes."""
        # Create chain: Node A -> Node B -> Node C
        node_a = MockQueueNode("node_a")
        node_b = MockQueueNode("node_b")
        node_c = MockQueueNode("node_c")
        
        # Setup connections
        node_a.setup_outputs(["signal"])
        node_b.setup_inputs(required=["trigger"])
        node_b.setup_outputs(["signal"])
        node_c.setup_inputs(required=["trigger"])
        
        # Create trigger chain
        trigger_ab = asyncio.Queue()
        trigger_bc = asyncio.Queue()
        
        node_a.output_queues["signal"] = trigger_ab
        node_b.input_queues["trigger"] = trigger_ab
        node_b.output_queues["signal"] = trigger_bc
        node_c.input_queues["trigger"] = trigger_bc
        
        # Send initial signal
        initial_signal = {"step": 1, "ready": True}
        await node_a.output_queues["signal"].put(initial_signal)
        
        # Node B receives and forwards
        received_by_b = await node_b.input_queues["trigger"].get()
        forwarded_signal = {"step": 2, "ready": True, "from": "node_b"}
        await node_b.output_queues["signal"].put(forwarded_signal)
        
        # Node C receives final signal
        received_by_c = await node_c.input_queues["trigger"].get()
        
        assert received_by_b["step"] == 1
        assert received_by_c["step"] == 2
        assert received_by_c["from"] == "node_b"


class TestConcurrentExecution:
    """Test concurrent execution patterns."""
    
    @pytest.mark.queue_framework
    @pytest.mark.asyncio
    async def test_concurrent_node_execution(self):
        """Test concurrent execution of multiple nodes."""
        # Create multiple nodes
        nodes = [MockQueueNode(f"node_{i}") for i in range(3)]
        
        # Setup each node
        for node in nodes:
            node.setup_inputs(required=["input"])
            node.setup_outputs(["output"])
        
        # Create concurrent tasks
        async def process_node(node, input_data):
            await node.input_queues["input"].put(input_data)
            result = await node.compute(input_data=input_data)
            await node.output_queues["output"].put(result)
            return result
        
        # Run nodes concurrently
        tasks = [
            process_node(nodes[0], "data_0"),
            process_node(nodes[1], "data_1"), 
            process_node(nodes[2], "data_2")
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All nodes should complete
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, dict)
            assert "output" in result
    
    @pytest.mark.queue_framework
    @pytest.mark.asyncio
    async def test_producer_consumer_pattern(self):
        """Test producer-consumer execution pattern."""
        producer = MockQueueNode("producer")
        consumer = MockQueueNode("consumer")
        
        producer.setup_outputs(["data"])
        consumer.setup_inputs(required=["data"])
        consumer.setup_outputs(["result"])
        
        # Shared queue
        data_queue = asyncio.Queue(maxsize=5)
        result_queue = asyncio.Queue()
        
        producer.output_queues["data"] = data_queue
        consumer.input_queues["data"] = data_queue
        consumer.output_queues["result"] = result_queue
        
        # Producer coroutine
        async def produce_data():
            for i in range(3):
                data = f"item_{i}"
                await producer.output_queues["data"].put(data)
                await asyncio.sleep(0.01)  # Small delay
        
        # Consumer coroutine
        async def consume_data():
            results = []
            for i in range(3):
                data = await consumer.input_queues["data"].get()
                result = await consumer.compute(input_data=data)
                await consumer.output_queues["result"].put(result)
                results.append(result)
            return results
        
        # Run producer and consumer concurrently
        producer_task = asyncio.create_task(produce_data())
        consumer_task = asyncio.create_task(consume_data())
        
        await producer_task
        results = await consumer_task
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
    
    @pytest.mark.queue_framework
    @pytest.mark.asyncio
    async def test_backpressure_handling(self):
        """Test backpressure handling in queue system."""
        fast_producer = MockQueueNode("fast_producer")
        slow_consumer = MockQueueNode("slow_consumer")
        
        fast_producer.setup_outputs(["data"])
        slow_consumer.setup_inputs(required=["data"])
        
        # Small queue to test backpressure
        limited_queue = asyncio.Queue(maxsize=2)
        fast_producer.output_queues["data"] = limited_queue
        slow_consumer.input_queues["data"] = limited_queue
        
        # Track production and consumption
        produced_items = []
        consumed_items = []
        
        async def fast_produce():
            for i in range(5):
                item = f"item_{i}"
                try:
                    await asyncio.wait_for(
                        fast_producer.output_queues["data"].put(item),
                        timeout=0.5
                    )
                    produced_items.append(item)
                except asyncio.TimeoutError:
                    break  # Backpressure stops production
        
        async def slow_consume():
            for _ in range(3):  # Only consume 3 items
                await asyncio.sleep(0.1)  # Slow consumption
                item = await slow_consumer.input_queues["data"].get()
                consumed_items.append(item)
        
        # Run with backpressure
        await asyncio.gather(fast_produce(), slow_consume())
        
        # Producer should be limited by consumer speed
        assert len(produced_items) >= 2  # At least queue size
        assert len(consumed_items) == 3  # Consumer processed 3 items
        assert len(produced_items) <= 5  # Producer may be blocked


class TestErrorHandling:
    """Test error handling in queue framework."""
    
    @pytest.mark.queue_framework
    @pytest.mark.asyncio
    async def test_node_error_handling(self):
        """Test error handling within nodes."""
        class ErrorNode(MockQueueNode):
            async def compute(self, **kwargs):
                if kwargs.get("cause_error"):
                    raise ValueError("Simulated error")
                return await super().compute(**kwargs)
        
        error_node = ErrorNode("error_node")
        error_node.setup_inputs(required=["input"])
        error_node.setup_outputs(["output"])
        
        # Normal operation should work
        normal_result = await error_node.compute(input_data="normal")
        assert isinstance(normal_result, dict)
        
        # Error case should raise exception
        with pytest.raises(ValueError, match="Simulated error"):
            await error_node.compute(cause_error=True)
    
    @pytest.mark.queue_framework
    @pytest.mark.asyncio
    async def test_queue_timeout_handling(self):
        """Test timeout handling for queue operations."""
        node = MockQueueNode("test_node")
        node.setup_inputs(required=["input"])
        
        # Try to get from empty queue with timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                node.input_queues["input"].get(),
                timeout=0.1
            )
    
    @pytest.mark.queue_framework
    @pytest.mark.asyncio
    async def test_queue_full_timeout(self):
        """Test timeout when queue is full."""
        node = MockQueueNode("test_node")
        node.setup_outputs(["output"])
        
        # Create small queue and fill it
        small_queue = asyncio.Queue(maxsize=1)
        node.output_queues["output"] = small_queue
        
        # Fill queue
        await small_queue.put("item1")
        
        # Next put should timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                small_queue.put("item2"),
                timeout=0.1
            )


class TestQueueFrameworkIntegration:
    """Integration tests for complete queue framework."""
    
    @pytest.mark.queue_framework
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_training_loop_simulation(self):
        """Test simulation of training loop with queue coordination."""
        # Create nodes for training loop
        get_batch = MockQueueNode("get_batch")
        network = MockQueueNode("network") 
        loss = MockQueueNode("loss")
        training_step = MockQueueNode("training_step")
        
        # Setup connections
        get_batch.setup_outputs(["batch"])
        network.setup_inputs(required=["batch"])
        network.setup_outputs(["predictions"])
        loss.setup_inputs(required=["predictions"])
        loss.setup_outputs(["loss_value"])
        training_step.setup_inputs(required=["loss_value"])
        training_step.setup_outputs(["ready_signal"])
        get_batch.setup_inputs(optional=["trigger"])
        
        # Create queues for connections
        batch_queue = asyncio.Queue()
        pred_queue = asyncio.Queue()
        loss_queue = asyncio.Queue()
        trigger_queue = asyncio.Queue()
        
        # Connect nodes
        get_batch.output_queues["batch"] = batch_queue
        network.input_queues["batch"] = batch_queue
        network.output_queues["predictions"] = pred_queue
        loss.input_queues["predictions"] = pred_queue
        loss.output_queues["loss_value"] = loss_queue
        training_step.input_queues["loss_value"] = loss_queue
        training_step.output_queues["ready_signal"] = trigger_queue
        get_batch.input_queues["trigger"] = trigger_queue
        
        # Simulate training steps
        async def run_training_step():
            # Get batch generates initial batch
            batch_data = {"images": "mock_images", "labels": "mock_labels"}
            await get_batch.output_queues["batch"].put(batch_data)
            
            # Network processes batch
            batch = await network.input_queues["batch"].get()
            predictions = await network.compute(input_data=batch)
            await network.output_queues["predictions"].put(predictions)
            
            # Loss computes loss
            preds = await loss.input_queues["predictions"].get()
            loss_result = await loss.compute(input_data=preds)
            await loss.output_queues["loss_value"].put(loss_result)
            
            # Training step updates and signals ready
            loss_val = await training_step.input_queues["loss_value"].get()
            step_result = await training_step.compute(input_data=loss_val)
            ready_signal = {"signal_type": "ready", "step": 1}
            await training_step.output_queues["ready_signal"].put(ready_signal)
            
            # Get batch receives trigger for next iteration
            trigger = await get_batch.input_queues["trigger"].get()
            return trigger
        
        # Run training step
        final_trigger = await run_training_step()
        
        # Should complete successfully
        assert final_trigger is not None
        assert isinstance(final_trigger, dict)
        assert final_trigger["signal_type"] == "ready"
        assert final_trigger["step"] == 1
    
    @pytest.mark.queue_framework
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_queue_performance(self):
        """Test queue framework performance with many operations."""
        node = MockQueueNode("perf_test")
        node.setup_inputs(required=["input"])
        node.setup_outputs(["output"])
        
        # Test with many queue operations
        num_operations = 1000
        
        start_time = time.time()
        
        # Producer task
        async def produce():
            for i in range(num_operations):
                await node.input_queues["input"].put(f"item_{i}")
        
        # Consumer task
        async def consume():
            results = []
            for i in range(num_operations):
                item = await node.input_queues["input"].get()
                result = await node.compute(input_data=item)
                await node.output_queues["output"].put(result)
                results.append(result)
            return results
        
        # Run concurrently
        producer_task = asyncio.create_task(produce())
        consumer_task = asyncio.create_task(consume())
        
        await producer_task
        results = await consumer_task
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete in reasonable time
        assert elapsed < 5.0, f"Performance test took too long: {elapsed}s"
        assert len(results) == num_operations
        
        # Calculate throughput
        throughput = num_operations / elapsed
        assert throughput > 200, f"Throughput too low: {throughput} ops/sec"