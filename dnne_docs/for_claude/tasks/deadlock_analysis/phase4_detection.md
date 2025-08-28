# Phase 4: Deadlock Detection

## Objective
Implement algorithms to detect deadlocks and identify their root causes.

## Tasks

### 4.1 Wait-For Graph Construction
- [ ] Build directed graph of wait dependencies
- [ ] Node A → Node B if A is waiting for output from B
- [ ] Include both data and trigger dependencies
- [ ] Handle multi-input wait conditions

### 4.2 Deadlock Detection Algorithms
- [ ] Implement cycle detection (DFS-based)
- [ ] Find strongly connected components
- [ ] Identify minimal deadlock cycles
- [ ] Detect partial deadlocks (subset of nodes stuck)

### 4.3 Starvation Detection
- [ ] Identify nodes waiting for multiple inputs
- [ ] Check if some inputs will never arrive
- [ ] Detect missing bootstrap signals
- [ ] Find orphaned data (produced but never consumed)

### 4.4 Synchronization Analysis
- [ ] Analyze barrier synchronization patterns
- [ ] Check for mismatched trigger/data timing
- [ ] Identify race conditions
- [ ] Detect split-join synchronization issues

## Implementation Notes

```python
class DeadlockDetector:
    def __init__(self, graph_model, pending_data):
        self.graph = graph_model
        self.pending_data = pending_data
        self.wait_for_graph = {}
        
    def build_wait_for_graph(self):
        """Build graph of who waits for whom"""
        self.wait_for_graph = {}
        
        for node in self.graph.nodes.values():
            if node.state == 'WAITING':
                waiting_for = self.get_nodes_waiting_for(node)
                if waiting_for:
                    self.wait_for_graph[node.id] = waiting_for
                    
    def get_nodes_waiting_for(self, node):
        """Determine which nodes this node is waiting for"""
        waiting_for = set()
        
        # Check each required input
        for input_name in node.inputs_required:
            if input_name not in node.inputs_available:
                # Find who should produce this input
                producers = self.graph.get_producers_for(node.id, input_name)
                waiting_for.update(producers)
                
        return waiting_for
        
    def find_cycles(self):
        """Find all cycles in wait-for graph using DFS"""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            if node in self.wait_for_graph:
                for neighbor in self.wait_for_graph[node]:
                    if neighbor not in visited:
                        if dfs(neighbor, path):
                            return True
                    elif neighbor in rec_stack:
                        # Found cycle
                        cycle_start = path.index(neighbor)
                        cycle = path[cycle_start:] + [neighbor]
                        cycles.append(cycle)
                        
            path.pop()
            rec_stack.remove(node)
            return False
            
        for node in self.wait_for_graph:
            if node not in visited:
                dfs(node, [])
                
        return cycles
        
    def find_minimal_cycle(self, cycles):
        """Find the smallest cycle that explains the deadlock"""
        if not cycles:
            return None
            
        # Sort by length and return shortest
        return min(cycles, key=len)
        
    def analyze_starvation(self):
        """Check for nodes that will never get all inputs"""
        starved_nodes = []
        
        for node in self.graph.nodes.values():
            if node.state == 'WAITING':
                # Check if node is waiting for inputs that will never come
                missing_inputs = self.get_missing_inputs(node)
                if missing_inputs and not self.can_eventually_receive(node, missing_inputs):
                    starved_nodes.append({
                        'node': node.id,
                        'missing': missing_inputs,
                        'reason': self.get_starvation_reason(node, missing_inputs)
                    })
                    
        return starved_nodes
```

## Detection Patterns

### Pattern 1: Simple Circular Wait
```
A → B → C → A
```
Each node waiting for the next in cycle.

### Pattern 2: Barrier Deadlock
```
Barrier waiting for trigger
Trigger producer waiting for data
Data producer waiting for barrier output
```

### Pattern 3: Multi-Input Starvation
```
Concat waiting for inputs [A, B, C]
A available, B available
C will never be produced (producer deadlocked)
```

### Pattern 4: Shadow Environment Cycle
```
IsaacGym → Observation → Network → Action → IsaacGym
With barriers/optimizers creating synchronization deadlock
```

## Success Metrics
- Detects all cycles in Franka_Coop_Nodes
- Identifies minimal deadlock cycle
- Distinguishes deadlock from starvation
- Provides clear explanation of why deadlock occurred