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

### System Initialization Barrier

DNNE uses a system-wide initialization barrier to ensure all nodes are created and connected before any processing begins:

```python
class Global:
    """Global state management with initialization barrier"""
    
    def init_system_ready(self):
        """Initialize the system-wide ready event"""
        self._system_ready = asyncio.Event()
        self._registered_nodes = set()
        self._ready_nodes = set()
        self._connections_ready = False
        
    def register_node(self, node_id: str):
        """Called by nodes during __init__"""
        if self._system_ready is None:
            raise RuntimeError(f"Node {node_id} trying to register before system initialized!")
        self._registered_nodes.add(node_id)
        
    def report_node_ready(self, node_id: str):
        """Called by nodes when their tasks start"""
        self._ready_nodes.add(node_id)
        
    async def wait_for_system_ready(self):
        """Nodes wait here until all connections are established"""
        await self._system_ready.wait()
        
    def report_connections_ready(self):
        """GraphRunner calls this after wiring all connections"""
        self._connections_ready = True
        self._system_ready.set()  # Release all waiting nodes
```

#### Initialization Sequence

1. **Runner initialization**: `g.init_system_ready()` called before creating nodes
2. **Node creation**: Each node calls `g.register_node()` in `__init__`
3. **Task startup**: Nodes call `g.report_node_ready()` when tasks start
4. **Barrier wait**: All nodes wait at `g.wait_for_system_ready()`
5. **Connection wiring**: GraphRunner establishes all connections
6. **Barrier release**: `g.report_connections_ready()` releases all nodes
7. **Processing begins**: Nodes start their main processing loops

This prevents race conditions where nodes might start processing before their input connections are established.

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

### MultiWaiter Pattern

The MultiWaiter utility provides efficient handling of multiple input queues with different waiting strategies:

```python
class MultiWaiter:
    """Efficiently wait for data from multiple input queues"""
    
    def __init__(self, input_queues: Dict[str, asyncio.Queue], 
                 mode: str = "all", timeout: Optional[float] = None):
        """
        Args:
            input_queues: Dictionary of input name -> queue
            mode: "all" (wait for all inputs) or "any" (first available)
            timeout: Optional timeout in seconds
        """
        self.input_queues = input_queues
        self.mode = mode
        self.timeout = timeout
        self.listeners = {}  # Persistent listener tasks for "any" mode
        
    async def wait_for_data(self) -> Dict[str, Any]:
        """Wait for data based on mode"""
        if self.mode == "all":
            # Simple sequential wait - efficient for "all" mode
            results = {}
            for name, queue in self.input_queues.items():
                if self.timeout:
                    data = await asyncio.wait_for(queue.get(), self.timeout)
                else:
                    data = await queue.get()
                results[name] = data
            return results
            
        elif self.mode == "any":
            # Use persistent listeners to avoid task churn
            if not self.listeners:
                await self._start_listeners()
            
            # Wait for first result
            result = await self.result_queue.get()
            return result
```

#### Key Benefits

1. **Eliminates Task Churn**: Instead of creating/canceling tasks on every wait cycle, MultiWaiter uses persistent listener tasks for "any" mode
2. **Prevents Race Conditions**: Stable listeners eliminate timing windows that caused deadlocks
3. **Efficient Memory Usage**: No constant allocation/deallocation of task objects
4. **Simple Sequential for "all" Mode**: When waiting for all inputs, uses straightforward sequential waits

#### Usage in Nodes

```python
class ConcatNode(QueueNode):
    """Concatenates multiple inputs efficiently"""
    
    def __init__(self):
        super().__init__()
        self.multi_waiter = None
        
    def setup_inputs(self):
        # Create MultiWaiter with appropriate mode
        self.multi_waiter = MultiWaiter(
            self.input_queues,
            mode="any"  # Process as inputs arrive
        )
    
    async def compute(self):
        # Efficient wait for any available input
        inputs = await self.multi_waiter.wait_for_data()
        
        # Process whichever inputs are ready
        if "input_a" in inputs:
            self.cached_a = inputs["input_a"]
        if "input_b" in inputs:
            self.cached_b = inputs["input_b"]
            
        # Concatenate available tensors
        available = [t for t in [self.cached_a, self.cached_b] if t is not None]
        if available:
            return torch.cat(available, dim=1)
        return None
```

#### Implementation Details

The MultiWaiter solved a critical deadlock issue in complex workflows:

**Problem**: Constant task creation/cancellation in tight loops caused race conditions
```python
# OLD PATTERN - Causes deadlocks
while True:
    tasks = [asyncio.create_task(q.get()) for q in queues]
    done, pending = await asyncio.wait(tasks, return_when=FIRST_COMPLETED)
    # Cancel pending tasks - THIS CAUSES RACE CONDITIONS!
    for task in pending:
        task.cancel()
```

**Solution**: Persistent listeners eliminate the race window
```python
# NEW PATTERN - No deadlocks
async def _start_listeners(self):
    """Start persistent listener tasks once"""
    for name, queue in self.input_queues.items():
        listener = asyncio.create_task(
            self._listen_to_queue(name, queue)
        )
        self.listeners[name] = listener

async def _listen_to_queue(self, name, queue):
    """Persistent listener - never cancelled in normal operation"""
    while True:
        data = await queue.get()
        await self.result_queue.put({name: data})
```

This pattern has proven so effective it inadvertently fixed longstanding deadlocks in complex workflows like Franka cooperative control.

### One-Time Configuration Inputs Pattern

**CRITICAL RULE**: If a node needs to manually control when/how it reads from a queue (like one-time configuration inputs), it should NOT include that input in `setup_inputs()`. Instead, manually create the queue.

#### The Problem: Double-Getter Deadlock

When a node lists an input in `setup_inputs()` AND also manually reads from that queue, it creates two competing getters:

```python
# WRONG - Creates double-getter deadlock!
class TrainingStepNode(QueueNode):
    def __init__(self, node_id: str):
        super().__init__(node_id)
        # DON'T DO THIS - creates first getter via MultiWaiter
        self.setup_inputs(required=["loss"], optional=["optimizer"])
        
    async def run(self):
        # Second getter - competes with MultiWaiter!
        config = await self.get_config_inputs(["optimizer"])  # DEADLOCK!
        self.optimizer = config["optimizer"]
```

The MultiWaiter (created by `setup_inputs`) tries to get from the optimizer queue, AND `get_config_inputs` also tries to get from it. With only one optimizer message sent, one getter will block forever.

#### The Solution: Manual Queue Creation

For one-time configuration inputs, bypass `setup_inputs` and create the queue manually:

```python
# CORRECT - No double-getter!
class TrainingStepNode(QueueNode):
    def __init__(self, node_id: str):
        super().__init__(node_id)
        # Only list repeating inputs in setup_inputs
        self.setup_inputs(required=["loss"])
        
        # Manually create queue for one-time config
        from asyncio import Queue
        self.input_queues["optimizer"] = Queue(maxsize=1)
        
    async def run(self):
        # Now only ONE getter for optimizer queue
        config = await self.get_config_inputs(["optimizer"])
        self.optimizer = config["optimizer"]
        
        # Continue with normal MultiWaiter for loss inputs
        await super().run()
```

#### Common Patterns

**Pattern 1: One-time Config + Repeating Data**
```python
class ProcessorNode(QueueNode):
    def __init__(self, node_id):
        super().__init__(node_id)
        # Only repeating inputs in setup_inputs
        self.setup_inputs(required=["data"])
        
        # Manual queues for one-time configs
        self.input_queues["config"] = Queue(maxsize=1)
        self.input_queues["model"] = Queue(maxsize=1)
        
    async def run(self):
        # Get configs once
        configs = await self.get_config_inputs(["config", "model"])
        self.config = configs["config"]
        self.model = configs["model"]
        
        # Process data stream with MultiWaiter
        await super().run()
```

**Pattern 2: Mixed Trigger Types**
```python
class GetBatchNode(QueueNode):
    def __init__(self, node_id):
        super().__init__(node_id)
        # No setup_inputs - we handle everything manually
        self.setup_inputs(required=[])
        
        # Different queue sizes for different patterns
        self.input_queues["dataloader"] = Queue(maxsize=1)  # One-time
        self.input_queues["schema"] = Queue(maxsize=1)      # One-time
        self.input_queues["trigger"] = Queue(maxsize=10)    # Repeated
        
    async def run(self):
        # Get one-time configs
        self.dataloader = await self.input_queues["dataloader"].get()
        self.schema = await self.input_queues["schema"].get()
        
        # Process triggers
        while self.running:
            trigger = await self.input_queues["trigger"].get()
            outputs = await self.compute()
            # ...
```

#### Nodes That Use This Pattern

- **TrainingStepNode**: Receives optimizer once, processes loss stream
- **SGDOptimizerNode**: Receives model once, emits optimizer once
- **GetBatchNode**: Receives dataloader/schema once, processes trigger stream
- **IsaacGymSimNode**: Custom queue handling for action/reset signals

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

## Best Practices

### 1. Node Design

- Keep nodes focused on single responsibility
- Make nodes stateless when possible
- Use configuration instead of hardcoding
- Include comprehensive logging

### 2. Queue Usage

- Handle full/empty queues gracefully
- Monitor queue depths in production
- Use timeouts for reliability

### 3. Error Handling

- Never let nodes crash silently
- Provide meaningful error messages
- Implement retry logic appropriately

### 4. Performance

- Profile queue bottlenecks
- Batch operations when beneficial
- Minimize queue hops
- Use appropriate concurrency

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