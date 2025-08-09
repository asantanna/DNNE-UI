"""
DNNE-specific command line arguments.

This module contains all command-line arguments specific to DNNE (Distributed Neural Network Editor)
that extend the base ComfyUI functionality.
"""

import argparse


def add_dnne_arguments(parser: argparse.ArgumentParser):
    """
    Add DNNE-specific command line arguments to the parser.
    
    Args:
        parser: The argparse.ArgumentParser instance to add arguments to
    """
    
    # DNNE Agent Server arguments
    parser.add_argument(
        "--agent-server-terminal", 
        action="store_true", 
        help="Start DNNE Agent Server in a new terminal window for debugging."
    )
    
    parser.add_argument(
        "--no-agent-server", 
        action="store_true", 
        help="Don't start the DNNE Agent Server automatically."
    )
    
    parser.add_argument(
        "--stop-agent-server", 
        action="store_true", 
        help="Stop any running DNNE Agent Server before starting."
    )
    
    parser.add_argument(
        "--restart-agent-server", 
        action="store_true", 
        help="Stop and restart the DNNE Agent Server."
    )
    
    parser.add_argument(
        "--agent-server-args", 
        type=str, 
        help="Additional arguments to pass to the DNNE Agent Server (e.g., '--verbose')"
    )