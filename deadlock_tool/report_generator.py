"""
Report generator for DNNE deadlock analysis.
Creates human-readable reports from analysis results.
"""

import sys
from typing import List, Optional, TextIO


class ReportGenerator:
    """Generates analysis reports"""
    
    def __init__(self, data_parser, patterns, root_causes: List, verbose: bool = False):
        self.data_parser = data_parser
        self.patterns = patterns
        self.root_causes = root_causes
        self.verbose = verbose
    
    def generate(self, output: Optional[TextIO] = None):
        """Generate the complete report"""
        if output is None:
            output = sys.stdout
        
        self._print_header(output)
        self._print_summary(output)
        
        if self.verbose:
            self._print_event_breakdown(output)
            self._print_node_activity(output)
        
        self._print_root_causes(output)
        self._print_other_issues(output)
        self._print_recommendations(output)
    
    def _print_header(self, output: TextIO):
        """Print report header"""
        output.write("\nDNNE Deadlock Analysis Report\n")
        output.write("=" * 60 + "\n")
        output.write(f"Data from: {self.data_parser.data_dir}\n")
    
    def _print_summary(self, output: TextIO):
        """Print summary statistics"""
        start_time, end_time = self.data_parser.get_time_range()
        duration = end_time - start_time
        
        output.write(f"Time range: 0.00s - {duration:.2f}s\n")
        output.write(f"Total events: {len(self.data_parser.events):,}\n")
        output.write(f"Total nodes: {len(self.data_parser.graph)}\n")
        
        # Count nodes by status
        never_started = len(self.patterns.nodes_never_started)
        stuck_waiting = len(self.patterns.nodes_stuck_waiting)
        never_output = len(self.patterns.nodes_never_output)
        
        if never_started > 0:
            output.write(f"⚠️  {never_started} nodes never started\n")
        if stuck_waiting > 0:
            output.write(f"⚠️  {stuck_waiting} nodes stuck waiting\n")
        if never_output > 0:
            output.write(f"⚠️  {never_output} nodes never produced output\n")
    
    def _print_event_breakdown(self, output: TextIO):
        """Print event type breakdown (verbose mode)"""
        output.write("\nEvent Breakdown:\n")
        counts = self.data_parser.get_event_counts()
        for event_type, count in sorted(counts.items()):
            output.write(f"  {event_type}: {count}\n")
    
    def _print_node_activity(self, output: TextIO):
        """Print node activity details (verbose mode)"""
        output.write("\nNode Activity Details:\n")
        
        # Show nodes that never started
        if self.patterns.nodes_never_started:
            output.write("\nNodes that never started:\n")
            for node_id in sorted(self.patterns.nodes_never_started):
                class_name = self.data_parser.graph.get(node_id, {}).get("class", "Unknown")
                output.write(f"  • Node {node_id} ({class_name})\n")
        
        # Show nodes stuck waiting
        if self.patterns.nodes_stuck_waiting:
            output.write("\nNodes stuck waiting:\n")
            for node_id, wait_time in sorted(self.patterns.nodes_stuck_waiting.items(), 
                                            key=lambda x: x[1], reverse=True):
                class_name = self.data_parser.node_classes.get(node_id, "Unknown")
                status = self.data_parser.node_wait_status.get(node_id, "unknown")
                output.write(f"  • Node {node_id} ({class_name}): {status} for {wait_time:.1f}s\n")
    
    def _print_root_causes(self, output: TextIO):
        """Print identified root causes"""
        output.write("\n🔍 ROOT CAUSE ANALYSIS:\n")
        output.write("-" * 40 + "\n")
        
        if not self.root_causes:
            output.write("No clear root causes identified.\n")
            return
        
        for i, cause in enumerate(self.root_causes, 1):
            output.write(f"\n{i}. PRIMARY ISSUE: Node {cause.node_id} ({cause.node_class})\n")
            output.write(f"   Type: {cause.cause_type}\n")
            output.write(f"   Problem: {cause.description}\n")
            
            if cause.affected_nodes:
                output.write(f"   Blocks {len(cause.affected_nodes)} downstream nodes:\n")
                # Show first 5 affected nodes
                for affected in cause.affected_nodes[:5]:
                    affected_class = self.data_parser.graph.get(affected, {}).get("class", "Unknown")
                    output.write(f"     → {affected} ({affected_class})\n")
                if len(cause.affected_nodes) > 5:
                    output.write(f"     ... and {len(cause.affected_nodes) - 5} more\n")
            
            # Suggest fix based on cause type
            output.write("   💡 Suggested fix: ")
            if cause.cause_type == "missing_bootstrap":
                output.write("Provide required bootstrap inputs at workflow start\n")
            elif cause.cause_type == "missing_input":
                output.write("Ensure upstream nodes are producing required data\n")
            elif cause.cause_type == "circular_dependency":
                output.write("Break cycle with bootstrap data or restructure connections\n")
            elif cause.cause_type == "no_producer":
                output.write("Add data source or trigger for this node\n")
            elif cause.cause_type == "free_running_blocked":
                output.write("Check node configuration - free-running nodes should auto-start\n")
    
    def _print_other_issues(self, output: TextIO):
        """Print other issues not covered by root causes"""
        output.write("\n📋 DETAILED OBSERVATIONS:\n")
        output.write("-" * 40 + "\n")
        
        # Section 1: Nodes that never received ANY input
        never_received_any = []
        for node_id, missing in self.patterns.nodes_never_received_input.items():
            # Check if node never received ANY of its expected inputs
            node_class = self.data_parser.graph.get(node_id, {}).get("class", "Unknown")
            # Get expected inputs for this node
            expected_count = len([c for c in self.data_parser.connections if len(c) >= 4 and c[2] == node_id])
            missing_count = len(missing)
            if expected_count > 0 and missing_count == expected_count:
                never_received_any.append((node_id, node_class, missing))
        
        if never_received_any:
            output.write("\n⚠️  NODES THAT NEVER RECEIVED INPUT:\n")
            for node_id, class_name, missing in never_received_any:
                output.write(f"• Node {node_id} ({class_name}) - expected: {', '.join(missing)}\n")
        
        # Section 2: Nodes that never produced output
        if self.patterns.nodes_never_output:
            output.write("\n⚠️  NODES THAT NEVER PRODUCED OUTPUT:\n")
            for node_id in sorted(self.patterns.nodes_never_output):
                class_name = self.data_parser.graph.get(node_id, {}).get("class", "Unknown")
                # Check if this node ran at all
                ran = node_id not in self.patterns.nodes_never_started
                if ran:
                    output.write(f"• Node {node_id} ({class_name}) - ran but produced no output\n")
                else:
                    output.write(f"• Node {node_id} ({class_name}) - never started\n")
        
        # Section 3: Stuck nodes (waiting for a long time)
        stuck_nodes = []
        for node_id, wait_time in self.patterns.nodes_stuck_waiting.items():
            class_name = self.data_parser.graph.get(node_id, {}).get("class", "Unknown")
            missing = self.patterns.nodes_never_received_input.get(node_id, [])
            stuck_nodes.append((node_id, class_name, missing, wait_time))
        
        if stuck_nodes:
            output.write("\n⚠️  STUCK NODES (potential deadlock):\n")
            for node_id, class_name, missing, wait_time in stuck_nodes:
                if missing:
                    output.write(f"• Node {node_id} ({class_name}) stuck {wait_time:.1f}s waiting for: {', '.join(missing)}\n")
                else:
                    output.write(f"• Node {node_id} ({class_name}) stuck {wait_time:.1f}s\n")
        
        # Section 4: Nodes currently waiting (but not stuck)
        waiting_nodes = []
        for node_id, missing in self.patterns.nodes_never_received_input.items():
            # Skip if already covered above
            if any(node_id == n[0] for n in never_received_any):
                continue
            if node_id in self.patterns.nodes_stuck_waiting:
                continue
            if any(rc.node_id == node_id for rc in self.root_causes):
                continue
                
            class_name = self.data_parser.graph.get(node_id, {}).get("class", "Unknown")
            waiting_nodes.append((node_id, class_name, missing))
        
        if waiting_nodes:
            output.write("\nℹ️  WAITING FOR DATA (normal operation):\n")
            for node_id, class_name, missing in waiting_nodes:
                output.write(f"• Node {node_id} ({class_name}) awaiting: {', '.join(missing)}\n")
        
        if not never_received_any and not self.patterns.nodes_never_output and not stuck_nodes and not waiting_nodes:
            output.write("No additional observations.\n")
    
    def _print_recommendations(self, output: TextIO):
        """Print generic recommendations"""
        output.write("\n💡 RECOMMENDATIONS:\n")
        output.write("-" * 40 + "\n")
        
        recommendations = []
        
        # Specific recommendations based on root causes
        if self.root_causes:
            primary_cause = self.root_causes[0]
            if primary_cause.cause_type == "missing_bootstrap":
                recommendations.append(
                    f"1. Provide bootstrap inputs for {primary_cause.node_class}: {primary_cause.description}"
                )
            elif primary_cause.cause_type == "missing_input":
                recommendations.append(
                    f"1. Ensure node {primary_cause.node_id} receives required inputs to start"
                )
            elif primary_cause.cause_type == "circular_dependency":
                recommendations.append(
                    "1. Break circular dependencies with bootstrap data or restructure workflow"
                )
            elif primary_cause.cause_type == "free_running_blocked":
                recommendations.append(
                    f"1. Debug {primary_cause.node_class} configuration - should start automatically"
                )
        
        # Check for synchronization nodes
        has_sync_nodes = any(
            "barrier" in str(self.data_parser.graph.get(nid, {}).get("class", "")).lower() or
            "eat_n" in str(self.data_parser.graph.get(nid, {}).get("class", "")).lower()
            for nid in self.data_parser.graph
        )
        
        if has_sync_nodes:
            recommendations.append(
                "2. Verify synchronization nodes (Barrier/Eat_N) have proper trigger patterns"
            )
        
        # Generic recommendations
        recommendations.extend([
            "3. Review the visual workflow for missing connections",
            "4. Check that all required node parameters are configured",
            "5. Consider adding debug logging to track data flow"
        ])
        
        for rec in recommendations:
            output.write(f"  {rec}\n")