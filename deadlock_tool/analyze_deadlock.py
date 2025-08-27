#!/usr/bin/env python3
"""
DNNE Deadlock Analysis Tool - Main Entry Point

Analyzes deadlock data collected with --debug-deadlock flag to identify
root causes of stuck workflows.

Usage:
    python analyze_deadlock.py [--data-dir /path/to/data] [--verbose]
"""

import argparse
import sys
from pathlib import Path

from data_parser import DeadlockDataParser
from pattern_analyzer import PatternAnalyzer
from root_cause_analyzer import RootCauseAnalyzer
from report_generator import ReportGenerator


def main():
    """Main entry point for deadlock analysis"""
    parser = argparse.ArgumentParser(description="Analyze DNNE deadlock data")
    parser.add_argument("--data-dir", default="/tmp/dnne_deadlock_data",
                       help="Path to deadlock data directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed analysis")
    parser.add_argument("--output", "-o", help="Output file for report (default: stdout)")
    args = parser.parse_args()
    
    # Parse the raw data
    data_parser = DeadlockDataParser(args.data_dir)
    if not data_parser.load_data():
        print(f"❌ Failed to load data from {args.data_dir}")
        sys.exit(1)
    
    # Analyze patterns in the data
    pattern_analyzer = PatternAnalyzer(
        data_parser.events,
        data_parser.graph,
        data_parser.connections,
        data_parser.node_configs
    )
    patterns = pattern_analyzer.analyze()
    
    # Find root causes
    root_cause_analyzer = RootCauseAnalyzer(
        data_parser.events,
        data_parser.graph,
        data_parser.connections,
        patterns
    )
    root_causes = root_cause_analyzer.find_root_causes()
    
    # Generate report
    report_gen = ReportGenerator(
        data_parser,
        patterns,
        root_causes,
        verbose=args.verbose
    )
    
    if args.output:
        with open(args.output, 'w') as f:
            report_gen.generate(f)
        print(f"✅ Report saved to {args.output}")
    else:
        report_gen.generate()


if __name__ == "__main__":
    main()