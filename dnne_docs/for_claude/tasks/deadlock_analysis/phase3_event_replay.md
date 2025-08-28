# Phase 3: Event Replay System

## Objective
Implement a system that replays logged events to simulate dataflow and track node state transitions.

## Tasks

### 3.1 Event Processing
- [ ] Parse event log JSON format
- [ ] Sort events by timestamp
- [ ] Handle different event types:
  - [ ] `QUEUE_PUT` - Node produced output
  - [ ] `QUEUE_GET_WAIT` - Node waiting for input
  - [ ] `QUEUE_GET_SUCCESS` - Node received input
  - [ ] `QUEUE_PUT_BLOCKED` - Output queue full
- [ ] Map events to graph model nodes

### 3.2 Pending Data Tracking
- [ ] Track data waiting at each connection
- [ ] Model queue capacity limits
- [ ] Handle data consumption order (FIFO)
- [ ] Track data that's been produced but not consumed

### 3.3 State Transition Engine
- [ ] Update node states based on events
- [ ] Check if nodes become ready after receiving input
- [ ] Handle execution completion events
- [ ] Model execution time (if available)

### 3.4 Progress Monitoring
- [ ] Track last progress timestamp
- [ ] Detect when no nodes are making progress
- [ ] Identify nodes that are permanently blocked
- [ ] Calculate time since last successful operation

## Implementation Notes

```python
class EventReplayEngine:
    def __init__(self, graph_model):
        self.graph = graph_model
        self.current_time = 0
        self.pending_data = {}  # (source, output, target, input) -> data_item
        self.event_history = []
        self.last_progress = 0
        
    def replay_events(self, events):
        sorted_events = sorted(events, key=lambda e: e['timestamp'])
        
        for event in sorted_events:
            self.current_time = event['timestamp']
            self.process_event(event)
            
            if self.is_progress_stalled():
                return self.analyze_deadlock()
                
    def process_event(self, event):
        if event['event_type'] == 'QUEUE_PUT':
            self.handle_output_produced(event)
        elif event['event_type'] == 'QUEUE_GET_SUCCESS':
            self.handle_input_consumed(event)
        elif event['event_type'] == 'QUEUE_GET_WAIT':
            self.handle_node_waiting(event)
            
    def handle_output_produced(self, event):
        source_node = event['node_id']
        output_name = event['output_name']
        
        # Find all connections from this output
        for conn in self.graph.get_connections_from(source_node, output_name):
            target_node, input_name = conn
            self.pending_data[(source_node, output_name, target_node, input_name)] = {
                'timestamp': event['timestamp'],
                'data': event.get('data_info', {})
            }
            
        self.last_progress = self.current_time
        
    def is_progress_stalled(self):
        # Check if no progress for significant time
        return (self.current_time - self.last_progress) > DEADLOCK_TIMEOUT
```

## Test Scenarios

### Scenario 1: Simple Pipeline
- A produces → B consumes → B produces → C consumes
- Verify correct state transitions

### Scenario 2: Barrier Release
- Data arrives at barrier
- Trigger arrives at barrier
- Barrier releases data
- Verify both conditions required

### Scenario 3: Eat_N Behavior
- First N inputs consumed
- Switches to passthrough mode
- Subsequent inputs passed through
- Verify mode transition

### Scenario 4: Circular Dependency
- A waits for B
- B waits for C
- C waits for A
- Detect deadlock

## Success Metrics
- Accurately tracks data flow through graph
- Node states match actual execution
- Detects when progress stops
- Identifies exact time of deadlock