"""
Root cause analyzer for DNNE deadlock analysis.
Traces back through dependency chains to find the actual source of blockages.
Uses node behavior knowledge base to understand node requirements.
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class RootCause:
    """Describes a root cause of deadlock"""
    node_id: str
    node_class: str
    cause_type: str  # "missing_input", "circular_dependency", "no_producer"
    description: str
    affected_nodes: List[str]  # Nodes blocked by this root cause
    severity: int  # 1-10, higher = more critical


class RootCauseAnalyzer:
    """Identifies root causes by tracing dependency chains with node behavior awareness"""
    
    def __init__(self, events: List[Dict], graph: Dict, connections: List, patterns):
        self.events = events
        self.graph = graph
        self.connections = connections
        self.patterns = patterns
        
        # Load node behavior knowledge base
        self.node_behaviors = self._load_node_behaviors()
        
        # Build dependency relationships
        self.dependencies = self._build_dependency_graph()
        self.reverse_dependencies = self._build_reverse_dependencies()
    
    def _load_node_behaviors(self) -> Dict:
        """Load node behavior definitions from JSON knowledge base"""
        behaviors_file = Path(__file__).parent / "node_behaviors.json"
        if behaviors_file.exists():
            with open(behaviors_file, 'r') as f:
                return json.load(f)
        return {"node_types": {}, "default_behavior": {}}
    
    def find_root_causes(self) -> List[RootCause]:
        """Find the root causes of all blockages"""
        root_causes = []
        
        # First, identify nodes that need bootstrap based on behaviors
        bootstrap_causes = self._find_bootstrap_dependent_nodes()
        root_causes.extend(bootstrap_causes)
        
        # Analyze nodes that never received input
        for node_id, missing_inputs in self.patterns.nodes_never_received_input.items():
            # Skip if already identified as bootstrap-dependent
            if any(rc.node_id == node_id for rc in bootstrap_causes):
                continue
            root_cause = self._trace_missing_input_cause(node_id, missing_inputs)
            if root_cause:
                root_causes.append(root_cause)
        
        # Analyze circular dependencies
        for cycle in self.patterns.circular_dependencies:
            root_cause = self._analyze_cycle(cycle)
            if root_cause:
                root_causes.append(root_cause)
        
        # Deduplicate and prioritize root causes
        root_causes = self._deduplicate_and_prioritize(root_causes)
        
        return root_causes
    
    def _get_node_behavior(self, node_class: str) -> Dict:
        """Get behavior definition for a node class"""
        node_types = self.node_behaviors.get("node_types", {})
        
        # Direct match
        if node_class in node_types:
            return node_types[node_class]
        
        # Try removing node ID suffix (e.g., "IsaacGymSimNode_25" -> "IsaacGymSimNode")
        if "_" in node_class:
            base_class = node_class.rsplit("_", 1)[0]
            if base_class in node_types:
                return node_types[base_class]
        
        # Try without "Node" suffix
        if node_class.endswith("Node") and node_class[:-4] in node_types:
            return node_types[node_class[:-4]]
        
        # Return default behavior
        return self.node_behaviors.get("default_behavior", {})
    
    def _is_free_running(self, node_class: str) -> bool:
        """Check if a node is free-running (doesn't need input)"""
        behavior = self._get_node_behavior(node_class)
        return behavior.get("category") == "free_running"
    
    def _is_bootstrap_dependent(self, node_class: str) -> bool:
        """Check if a node needs bootstrap inputs to start"""
        behavior = self._get_node_behavior(node_class)
        return behavior.get("category") in ["bootstrap_dependent", "bootstrap_dependent_provider"]
    
    def _get_bootstrap_inputs(self, node_class: str) -> List[str]:
        """Get bootstrap inputs required by a node"""
        behavior = self._get_node_behavior(node_class)
        return behavior.get("bootstrap_inputs", [])
    
    def _find_bootstrap_dependent_nodes(self) -> List[RootCause]:
        """Identify nodes that need bootstrap and haven't received it"""
        bootstrap_causes = []
        
        # Check ALL bootstrap-dependent nodes, not just those that never started
        for node_id, node_info in self.graph.items():
            node_class = node_info.get("class", "Unknown")
            
            if self._is_bootstrap_dependent(node_class):
                bootstrap_inputs = self._get_bootstrap_inputs(node_class)
                
                if bootstrap_inputs:
                    # Check if node received its bootstrap inputs in the event log
                    received_bootstrap = False
                    for event in self.events:
                        if (event.get("type") == "QUEUE_GET_SUCCESS" and
                            event.get("node") == node_id and
                            any(inp in event.get("queue", "") for inp in bootstrap_inputs)):
                            received_bootstrap = True
                            break
                    
                    if not received_bootstrap:
                        # This node needs bootstrap but didn't get it
                        affected = self._calculate_affected_nodes(node_id)
                        
                        # Check if node started anyway (shouldn't happen, but let's be thorough)
                        node_started = node_id not in self.patterns.nodes_never_started
                        if node_started:
                            description = f"WARNING: Node started without receiving bootstrap inputs: {', '.join(bootstrap_inputs)}"
                            severity = 10  # This is a serious issue if it happens
                        else:
                            description = f"Node needs bootstrap inputs: {', '.join(bootstrap_inputs)} to start"
                            severity = 8 + (len(affected) // 5)
                        
                        bootstrap_causes.append(RootCause(
                            node_id=node_id,
                            node_class=node_class,
                            cause_type="missing_bootstrap",
                            description=description,
                            affected_nodes=affected,
                            severity=severity
                        ))
        
        return bootstrap_causes
    
    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """Build graph of what each node depends on"""
        deps = {}
        for conn in self.connections:
            if len(conn) >= 4:
                from_node = conn[0]
                to_node = conn[2]
                if to_node not in deps:
                    deps[to_node] = set()
                deps[to_node].add(from_node)
        return deps
    
    def _build_reverse_dependencies(self) -> Dict[str, Set[str]]:
        """Build graph of what depends on each node"""
        rev_deps = {}
        for conn in self.connections:
            if len(conn) >= 4:
                from_node = conn[0]
                to_node = conn[2]
                if from_node not in rev_deps:
                    rev_deps[from_node] = set()
                rev_deps[from_node].add(to_node)
        return rev_deps
    
    def _trace_missing_input_cause(self, node_id: str, missing_inputs: List[str]) -> Optional[RootCause]:
        """Trace back why a node didn't receive its inputs, using node behavior knowledge"""
        # Find what nodes should have provided these inputs
        providers = {}
        for conn in self.connections:
            if len(conn) >= 4 and conn[2] == node_id and conn[3] in missing_inputs:
                provider = conn[0]
                input_name = conn[3]
                providers[input_name] = provider
        
        # Check if providers themselves are blocked
        blocked_providers = []
        free_running_blocked = []
        
        for input_name, provider_id in providers.items():
            provider_class = self.graph.get(provider_id, {}).get("class", "Unknown")
            
            if provider_id in self.patterns.nodes_never_started:
                blocked_providers.append(provider_id)
                # Check if this is a free-running node that's blocked
                if self._is_free_running(provider_class):
                    free_running_blocked.append(provider_id)
        
        # If a free-running node is blocked, that's likely a configuration issue
        if free_running_blocked:
            blocker = free_running_blocked[0]
            affected = self._calculate_affected_nodes(blocker)
            
            return RootCause(
                node_id=blocker,
                node_class=self.graph.get(blocker, {}).get("class", "Unknown"),
                cause_type="free_running_blocked",
                description=f"Free-running node failed to start - check configuration",
                affected_nodes=affected,
                severity=9  # High severity - free-running nodes should always work
            )
        
        if blocked_providers:
            # Recursively trace back to find the ultimate blocker
            ultimate_blocker = self._find_ultimate_blocker(blocked_providers[0])
            
            if ultimate_blocker:
                blocker_class = self.graph.get(ultimate_blocker, {}).get("class", "Unknown")
                affected = self._calculate_affected_nodes(ultimate_blocker)
                
                # Determine cause type based on node behavior
                if self._is_bootstrap_dependent(blocker_class):
                    bootstrap_inputs = self._get_bootstrap_inputs(blocker_class)
                    description = f"Node needs bootstrap inputs: {', '.join(bootstrap_inputs)} to start"
                    cause_type = "missing_bootstrap"
                    severity = 8
                elif ultimate_blocker in self.patterns.nodes_never_received_input:
                    missing = self.patterns.nodes_never_received_input[ultimate_blocker]
                    description = f"Node needs input on: {', '.join(missing)} to start the workflow"
                    cause_type = "missing_input"
                    severity = 7
                else:
                    description = f"Node has no way to start - needs bootstrap or initial trigger"
                    cause_type = "no_producer"
                    severity = 6
                
                return RootCause(
                    node_id=ultimate_blocker,
                    node_class=blocker_class,
                    cause_type=cause_type,
                    description=description,
                    affected_nodes=affected,
                    severity=severity + (len(affected) // 10)
                )
        
        return None
    
    def _find_ultimate_blocker(self, node_id: str, visited: Set[str] = None) -> Optional[str]:
        """Recursively find the ultimate blocking node"""
        if visited is None:
            visited = set()
        
        if node_id in visited:
            return None  # Circular dependency
        visited.add(node_id)
        
        # If this node has no dependencies, it's a root
        if node_id not in self.dependencies or not self.dependencies[node_id]:
            return node_id
        
        # If this node's dependencies are all satisfied, it's the blocker
        deps = self.dependencies.get(node_id, set())
        blocked_deps = []
        
        for dep in deps:
            if dep in self.patterns.nodes_never_started or dep in self.patterns.nodes_never_received_input:
                blocked_deps.append(dep)
        
        if not blocked_deps:
            # All dependencies ran, so this node is the problem
            return node_id
        
        # Recursively check blocked dependencies
        for dep in blocked_deps:
            ultimate = self._find_ultimate_blocker(dep, visited)
            if ultimate:
                return ultimate
        
        return node_id
    
    def _calculate_affected_nodes(self, blocker_id: str) -> List[str]:
        """Calculate all nodes affected by a blocker"""
        affected = set()
        to_check = [blocker_id]
        
        while to_check:
            current = to_check.pop(0)
            if current in affected:
                continue
            affected.add(current)
            
            # Add nodes that depend on current
            dependents = self.reverse_dependencies.get(current, set())
            to_check.extend(dependents)
        
        # Remove the blocker itself from affected list
        affected.discard(blocker_id)
        return sorted(list(affected))
    
    def _analyze_cycle(self, cycle: List[str]) -> RootCause:
        """Analyze a circular dependency cycle"""
        # Find which node in the cycle would need bootstrap to break it
        # Usually it's the node that produces something others consume
        
        # Count how many nodes depend on each cycle member
        dependency_counts = {}
        for node in cycle:
            count = len(self.reverse_dependencies.get(node, set()))
            dependency_counts[node] = count
        
        # The node with most dependents is likely the best bootstrap point
        bootstrap_node = max(cycle, key=lambda n: dependency_counts[n])
        
        affected = []
        for node in cycle:
            affected.extend(self.reverse_dependencies.get(node, []))
        affected = list(set(affected))
        
        return RootCause(
            node_id=bootstrap_node,
            node_class=self.graph.get(bootstrap_node, {}).get("class", "Unknown"),
            cause_type="circular_dependency",
            description=f"Part of circular dependency cycle: {' → '.join(cycle)}",
            affected_nodes=affected,
            severity=10  # Circular dependencies are critical
        )
    
    def _deduplicate_and_prioritize(self, root_causes: List[RootCause]) -> List[RootCause]:
        """Remove duplicate root causes and sort by severity"""
        # Remove duplicates by node_id
        seen = {}
        for cause in root_causes:
            if cause.node_id not in seen or cause.severity > seen[cause.node_id].severity:
                seen[cause.node_id] = cause
        
        # Sort by severity (descending) and number of affected nodes
        unique_causes = list(seen.values())
        unique_causes.sort(key=lambda c: (c.severity, len(c.affected_nodes)), reverse=True)
        
        return unique_causes