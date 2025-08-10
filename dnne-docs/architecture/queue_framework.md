# Queue Framework Architecture

## Overview

The DNNE Queue Framework provides the async, event-driven runtime for exported workflows. Inspired by ROS (Robot Operating System) and dataflow architectures, it enables real-time performance crucial for robotics and ML applications.

## Core Concepts

### Async Queue-Based Communication

All nodes communicate through async queues:

```python
# Node A sends data
await output_queue.put(data)

# Node B receives data
data = await input_queue.get()
```

Benefits:
- **Non-blocking**: Nodes never wait synchronously
- **Backpressure**: Automatic flow control
- **Decoupled**: Nodes are independent
- **Scalable**: Easy to parallelize

### Event-Driven Execution

Nodes react to incoming data:

```python
async def process(self):
    while True:
        # Wait for event (data arrival)
        input_data = await self.get_input()
        
        # Process event
        output_data = self.compute(input_data)
        
        # Trigger downstream events
        await self.send_output(output_data)
```

## Framework Components

### QueueNode Base Class

The foundation for all nodes:

```python
class QueueNode:
    def __init__(self):
        self.input_queues = {}
        self.output_queues = {}
        self.running = False
        
    async def get_input(self, input_name="default", timeout=None):
        """Get data from input queue"""
        queue = self.input_queues[input_name]
        if timeout:
            return await asyncio.wait_for(queue.get(), timeout)
        return await queue.get()
    
    async def send_output(self, data, output_name="default"):
        """Send data to all connected output queues"""
        if output_name in self.output_queues:
            for queue in self.output_queues[output_name]:
                await queue.put(data)
    
    async def process(self):
        """Override this in subclasses"""
        raise NotImplementedError
    
    async def run(self):
        """Main execution loop"""
        self.running = True
        try:
            await self.process()
        except Exception as e:
            logger.error(f"Node error: {e}")
            self.running = False
```

### GraphRunner

Orchestrates node execution:

```python
class GraphRunner:
    def __init__(self, nodes):
        self.nodes = nodes
        self.tasks = []
        
    async def run(self):
        """Start all nodes concurrently"""
        # Create tasks for each node
        for node in self.nodes:
            task = asyncio.create_task(node.run())
            self.tasks.append(task)
        
        # Wait for all tasks
        try:
            await asyncio.gather(*self.tasks)
        except KeyboardInterrupt:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
```

### Queue Wiring

Connecting nodes together:

```python
def wire_nodes(connections):
    """Wire nodes based on connection list"""
    for source, output_name, target, input_name in connections:
        # Create queue for this connection
        queue = asyncio.Queue(maxsize=100)
        
        # Connect source output to queue
        if output_name not in source.output_queues:
            source.output_queues[output_name] = []
        source.output_queues[output_name].append(queue)
        
        # Connect queue to target input
        target.input_queues[input_name] = queue
```

## Design Patterns

### Producer-Consumer Pattern

Common in data processing:

```python
class DataProducer(QueueNode):
    async def process(self):
        while True:
            data = await self.generate_data()
            await self.send_output(data)
            await asyncio.sleep(0.1)  # Rate limiting

class DataConsumer(QueueNode):
    async def process(self):
        while True:
            data = await self.get_input()
            await self.process_data(data)
```

### Request-Response Pattern

For synchronous-style operations:

```python
class RequestNode(QueueNode):
    async def process(self):
        request_id = 0
        while True:
            # Send request with ID
            await self.send_output({
                'id': request_id,
                'data': await self.get_input()
            })
            
            # Wait for response
            response = await self.get_input('response')
            if response['id'] == request_id:
                await self.send_output(response['result'])
            request_id += 1
```

### Fork-Join Pattern

For parallel processing:

```python
class ForkNode(QueueNode):
    async def process(self):
        while True:
            data = await self.get_input()
            # Send to multiple outputs
            await self.send_output(data, 'branch1')
            await self.send_output(data, 'branch2')

class JoinNode(QueueNode):
    async def process(self):
        while True:
            # Wait for both inputs
            data1 = await self.get_input('input1')
            data2 = await self.get_input('input2')
            # Combine and send
            await self.send_output(combine(data1, data2))
```

### State Machine Pattern

For complex control flow:

```python
class StateMachineNode(QueueNode):
    def __init__(self):
        super().__init__()
        self.state = 'IDLE'
        
    async def process(self):
        while True:
            event = await self.get_input()
            
            if self.state == 'IDLE':
                if event.type == 'START':
                    self.state = 'RUNNING'
                    await self.send_output('started')
                    
            elif self.state == 'RUNNING':
                if event.type == 'STOP':
                    self.state = 'IDLE'
                    await self.send_output('stopped')
                else:
                    result = await self.process_event(event)
                    await self.send_output(result)
```

## Queue Management

### Queue Sizing

Preventing memory issues:

```python
# Bounded queues prevent runaway memory
queue = asyncio.Queue(maxsize=100)

# Handle full queues
try:
    await asyncio.wait_for(
        queue.put(item), 
        timeout=1.0
    )
except asyncio.TimeoutError:
    logger.warning("Queue full, dropping item")
```

### Priority Queues

For importance-based processing:

```python
import asyncio
import heapq

class PriorityQueue(asyncio.Queue):
    def _init(self, maxsize):
        self._queue = []
    
    def _put(self, item):
        heapq.heappush(self._queue, item)
    
    def _get(self):
        return heapq.heappop(self._queue)
```

### Queue Monitoring

Debugging and performance:

```python
class MonitoredQueue(asyncio.Queue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.put_count = 0
        self.get_count = 0
        
    async def put(self, item):
        self.put_count += 1
        await super().put(item)
        
    async def get(self):
        self.get_count += 1
        return await super().get()
    
    def stats(self):
        return {
            'size': self.qsize(),
            'puts': self.put_count,
            'gets': self.get_count,
            'waiting': len(self._getters)
        }
```

## Performance Optimization

### Batching

Process multiple items together:

```python
class BatchProcessor(QueueNode):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        
    async def process(self):
        while True:
            batch = []
            # Collect batch
            for _ in range(self.batch_size):
                try:
                    item = await asyncio.wait_for(
                        self.get_input(), 
                        timeout=0.1
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
            
            if batch:
                # Process entire batch
                results = self.process_batch(batch)
                for result in results:
                    await self.send_output(result)
```

### Zero-Copy Operations

Minimize data copying:

```python
class ZeroCopyNode(QueueNode):
    async def process(self):
        while True:
            # Get reference, not copy
            tensor_ref = await self.get_input()
            
            # In-place operations
            tensor_ref.mul_(2.0)
            
            # Send reference
            await self.send_output(tensor_ref)
```

### CPU/GPU Optimization

Proper device management:

```python
class GPUNode(QueueNode):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = torch.device(device)
        
    async def process(self):
        while True:
            data = await self.get_input()
            
            # Move to GPU if needed
            if data.device != self.device:
                data = data.to(self.device)
            
            # GPU processing
            result = self.gpu_compute(data)
            
            await self.send_output(result)
```

## Error Handling

### Graceful Degradation

Continue operating despite errors:

```python
class ResilientNode(QueueNode):
    async def process(self):
        consecutive_errors = 0
        while True:
            try:
                data = await self.get_input()
                result = self.risky_operation(data)
                await self.send_output(result)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error: {e}")
                
                if consecutive_errors > 10:
                    logger.error("Too many errors, shutting down")
                    break
                    
                # Send default/safe output
                await self.send_output(self.get_safe_default())
```

### Circuit Breaker Pattern

Prevent cascading failures:

```python
class CircuitBreakerNode(QueueNode):
    def __init__(self, failure_threshold=5):
        super().__init__()
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.circuit_open = False
        
    async def process(self):
        while True:
            if self.circuit_open:
                await asyncio.sleep(5)  # Wait before retry
                self.circuit_open = False
                self.failure_count = 0
                
            try:
                data = await self.get_input()
                result = await self.external_call(data)
                await self.send_output(result)
                self.failure_count = 0
            except Exception as e:
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self.circuit_open = True
                    logger.error("Circuit breaker opened")
```

## Integration Examples

### With Isaac Gym

Real-time simulation integration:

```python
class IsaacGymStepNode(QueueNode):
    async def process(self):
        while True:
            # Wait for action
            action = await self.get_input('action')
            
            # Step simulation (CPU-bound)
            await asyncio.get_event_loop().run_in_executor(
                None,  # Default executor
                self.gym.step,
                action
            )
            
            # Get observations
            obs = self.gym.get_observations()
            rewards = self.gym.get_rewards()
            
            # Send results
            await self.send_output(obs, 'observations')
            await self.send_output(rewards, 'rewards')
```

### With Neural Networks

Async inference:

```python
class NeuralNetNode(QueueNode):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        
    async def process(self):
        while True:
            input_tensor = await self.get_input()
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
            
            await self.send_output(output)
```

## Best Practices

### 1. Node Design

- Keep nodes focused on single responsibility
- Make nodes stateless when possible
- Use configuration instead of hardcoding
- Include comprehensive logging

### 2. Queue Usage

- Set appropriate queue sizes
- Handle full/empty queues gracefully
- Monitor queue depths in production
- Use timeouts for reliability

### 3. Error Handling

- Never let nodes crash silently
- Provide meaningful error messages
- Use safe defaults when possible
- Implement retry logic appropriately

### 4. Performance

- Profile queue bottlenecks
- Batch operations when beneficial
- Minimize queue hops
- Use appropriate concurrency

## Debugging Tools

### Queue Visualizer

```python
class QueueVisualizer:
    def __init__(self, graph_runner):
        self.runner = graph_runner
        
    def print_status(self):
        for node in self.runner.nodes:
            print(f"\nNode: {node.__class__.__name__}")
            print(f"  Running: {node.running}")
            for name, queue in node.input_queues.items():
                print(f"  Input '{name}': {queue.qsize()} items")
            for name, queues in node.output_queues.items():
                print(f"  Output '{name}': {len(queues)} connections")
```

### Event Tracer

```python
class EventTracer:
    def __init__(self):
        self.events = []
        
    def log_event(self, node_id, event_type, data_shape=None):
        self.events.append({
            'timestamp': time.time(),
            'node_id': node_id,
            'event_type': event_type,
            'data_shape': data_shape
        })
    
    def analyze(self):
        # Compute throughput, latency, etc.
        pass
```

## Future Directions

### Planned Enhancements

1. **Distributed Queues**
   - Cross-machine communication
   - Cloud-native deployment
   - Fault tolerance

2. **Advanced Scheduling**
   - Priority-based execution
   - Resource-aware scheduling
   - Dynamic load balancing

3. **Monitoring Integration**
   - Prometheus metrics
   - OpenTelemetry tracing
   - Real-time dashboards

4. **Type Safety**
   - Runtime type checking
   - Schema validation
   - Automatic serialization

The Queue Framework provides the foundation for DNNE's real-time, scalable execution model, enabling complex ML and robotics workflows to run efficiently in production environments.