# Phase 5: Root Cause Analysis

## Objective
Generate clear, actionable explanations of deadlock root causes with suggested fixes.

## Tasks

### 5.1 Deadlock Chain Explanation
- [ ] Convert cycles into human-readable explanations
- [ ] Show the complete blocking chain
- [ ] Highlight the "weakest link" that could break the cycle
- [ ] Explain why each node is blocked

### 5.2 Root Cause Identification
- [ ] Distinguish primary cause from secondary effects
- [ ] Identify missing bootstrap signals
- [ ] Find synchronization mismatches
- [ ] Detect incorrect connection patterns

### 5.3 Fix Recommendations
- [ ] Suggest adding bootstrap signals where needed
- [ ] Recommend connection changes
- [ ] Propose barrier/trigger adjustments
- [ ] Identify nodes that could self-start

### 5.4 Report Generation
- [ ] Create structured analysis report
- [ ] Include visual representation (ASCII or graphical)
- [ ] Provide confidence scores for findings
- [ ] Generate both technical and summary views

## Implementation Notes

```python
class RootCauseAnalyzer:
    def __init__(self, detector_results, graph_model):
        self.cycles = detector_results['cycles']
        self.starvation = detector_results['starvation']
        self.graph = graph_model
        
    def analyze_cycle(self, cycle):
        """Generate detailed analysis of a deadlock cycle"""
        analysis = {
            'cycle': cycle,
            'blocking_chain': [],
            'root_cause': None,
            'fixes': []
        }
        
        # Build blocking chain explanation
        for i, node_id in enumerate(cycle[:-1]):  # Skip last (duplicate of first)
            next_node = cycle[i + 1] if i + 1 < len(cycle) else cycle[0]
            
            blocking = {
                'node': node_id,
                'state': self.graph.nodes[node_id].state,
                'waiting_for': self.explain_wait(node_id, next_node),
                'why_blocked': self.get_blocking_reason(node_id)
            }
            analysis['blocking_chain'].append(blocking)
            
        # Identify root cause
        analysis['root_cause'] = self.identify_root_cause(cycle)
        
        # Generate fix suggestions
        analysis['fixes'] = self.suggest_fixes(cycle, analysis['root_cause'])
        
        return analysis
        
    def explain_wait(self, waiting_node, blocking_node):
        """Explain what waiting_node needs from blocking_node"""
        node = self.graph.nodes[waiting_node]
        
        # Check data dependencies
        for input_name in node.inputs_required:
            producers = self.graph.get_producers_for(waiting_node, input_name)
            if blocking_node in producers:
                return f"input '{input_name}' from {blocking_node}"
                
        # Check trigger dependencies (for barriers)
        if isinstance(node, BarrierNode):
            trigger_producers = self.graph.get_trigger_producers(waiting_node)
            if blocking_node in trigger_producers:
                return f"trigger signal from {blocking_node}"
                
        return f"dependency from {blocking_node}"
        
    def identify_root_cause(self, cycle):
        """Identify the primary cause of the deadlock"""
        
        # Check for missing bootstrap
        bootstrap_nodes = ['SGDOptimizerNode', 'IsaacGymSimNode']
        for node_id in cycle:
            node = self.graph.nodes[node_id]
            if any(boot in node.node_class for boot in bootstrap_nodes):
                if not self.has_bootstrap_signal(node_id):
                    return {
                        'type': 'missing_bootstrap',
                        'node': node_id,
                        'description': f"{node_id} needs bootstrap signal to start the cycle"
                    }
                    
        # Check for barrier synchronization issue
        barriers_in_cycle = [n for n in cycle if 'Barrier' in self.graph.nodes[n].node_class]
        if barriers_in_cycle:
            return {
                'type': 'barrier_synchronization',
                'nodes': barriers_in_cycle,
                'description': "Barriers waiting for triggers that depend on barrier outputs"
            }
            
        # Default to circular dependency
        return {
            'type': 'circular_dependency',
            'description': "Pure circular wait with no self-starting nodes"
        }
        
    def suggest_fixes(self, cycle, root_cause):
        """Generate fix suggestions based on root cause"""
        fixes = []
        
        if root_cause['type'] == 'missing_bootstrap':
            fixes.append({
                'action': 'add_bootstrap',
                'target': root_cause['node'],
                'command': f"--override {root_cause['node']}:bootstrap=True",
                'description': f"Enable bootstrap signal for {root_cause['node']}"
            })
            
        elif root_cause['type'] == 'barrier_synchronization':
            fixes.append({
                'action': 'adjust_barriers',
                'description': "Consider using Eat_N instead of Barrier for async triggering"
            })
            fixes.append({
                'action': 'add_initial_trigger',
                'description': "Add a startup trigger signal to break the cycle"
            })
            
        # Always suggest breaking the weakest link
        weakest_link = self.find_weakest_link(cycle)
        if weakest_link:
            fixes.append({
                'action': 'break_connection',
                'source': weakest_link['from'],
                'target': weakest_link['to'],
                'description': f"Consider making {weakest_link['to']} independent of {weakest_link['from']}"
            })
            
        return fixes
        
    def generate_report(self, analysis):
        """Generate human-readable report"""
        report = []
        report.append("="*60)
        report.append("DEADLOCK ANALYSIS REPORT")
        report.append("="*60)
        report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append(f"  Root Cause: {analysis['root_cause']['description']}")
        report.append(f"  Cycle Length: {len(analysis['cycle']) - 1} nodes")
        report.append("")
        
        # Blocking chain
        report.append("BLOCKING CHAIN:")
        for block in analysis['blocking_chain']:
            report.append(f"  → {block['node']} ({block['state']})")
            report.append(f"    Waiting for: {block['waiting_for']}")
            report.append(f"    Reason: {block['why_blocked']}")
        report.append(f"  → {analysis['cycle'][0]} (cycle completes)")
        report.append("")
        
        # Suggested fixes
        report.append("SUGGESTED FIXES:")
        for i, fix in enumerate(analysis['fixes'], 1):
            report.append(f"  {i}. {fix['description']}")
            if 'command' in fix:
                report.append(f"     Command: {fix['command']}")
        report.append("")
        
        # ASCII visualization
        report.append("VISUAL REPRESENTATION:")
        report.append(self.generate_ascii_diagram(analysis['cycle']))
        
        return "\n".join(report)
```

## Report Examples

### Example 1: Missing Bootstrap
```
DEADLOCK ANALYSIS REPORT
========================

SUMMARY:
  Root Cause: SGDOptimizerNode_40 needs bootstrap signal to start the cycle
  Cycle Length: 7 nodes

BLOCKING CHAIN:
  → IsaacGymSimNode_25 (WAITING)
    Waiting for: input 'action' from ConcatNode_42
    Reason: Cannot step simulation without action
  → ConcatNode_42 (WAITING)
    Waiting for: input 'input_a' from NetworkNode_33
    Reason: Needs all inputs before concatenation
  → NetworkNode_33 (WAITING)
    Waiting for: input 'input' from BarrierNode_74
    Reason: Network requires input tensor
  → BarrierNode_74 (WAITING)
    Waiting for: trigger signal from SGDOptimizerNode_40
    Reason: Barrier holds data until triggered
  → SGDOptimizerNode_40 (WAITING)
    Waiting for: input 'loss' from CustomComputationNode_67
    Reason: Optimizer needs loss to compute gradients
  → CustomComputationNode_67 (WAITING)
    Waiting for: input 'input' from Eat_NNode_73
    Reason: Computation needs input data
  → Eat_NNode_73 (WAITING)
    Waiting for: input 'observation' from IsaacGymSimNode_25
    Reason: Waiting for first observation to consume
  → IsaacGymSimNode_25 (cycle completes)

SUGGESTED FIXES:
  1. Enable bootstrap signal for SGDOptimizerNode_40
     Command: --override SGDOptimizerNode_40:bootstrap=True
  2. Consider using Eat_N instead of Barrier for async triggering
  3. Add initial observation to break the cycle at Eat_NNode_73
```

## Success Metrics
- Clear explanation of why deadlock occurred
- Actionable fix suggestions
- Correct identification of root cause vs symptoms
- Report readable by non-experts