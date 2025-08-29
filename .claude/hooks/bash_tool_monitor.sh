#!/bin/bash
set -e  # Exit on any error

# Read the JSON input from Claude Code
json_input=$(cat)

# Parse with error checking
if ! tool_name=$(echo "$json_input" | jq -r '.tool_name' 2>/dev/null); then
    echo "ERROR: Failed to parse tool_name from JSON" >&2
    exit 1
fi

if ! command=$(echo "$json_input" | jq -r '.tool_input.command // empty' 2>/dev/null); then
    echo "ERROR: Failed to parse command from JSON" >&2
    exit 1
fi

# Check if it's trying to run graph_exporter.py (handles multi-line)
if [[ "$tool_name" == "Bash" ]] && [[ "$command" =~ python[0-9]* ]] && [[ "$command" =~ graph_exporter\.py ]]; then
    echo "BLOCKED: Don't use graph_exporter.py - it's deprecated!" >&2
    echo "Use claude_scripts/programmatic_export.py instead!" >&2
    exit 2
fi

# Check for stderr redirection followed by pipe
if [[ "$tool_name" == "Bash" ]] && [[ "$command" =~ (2>&1|&>|>&)[[:space:]]*\| ]]; then
    echo "BLOCKED: Incorrect syntax when redirecting stderr followed by a pipe." >&2
    echo "Instead, use parentheses: (command 2>&1) | next_command" >&2
    exit 2
fi

# Block ALL direct npm usage
if [[ "$tool_name" == "Bash" ]] && [[ "$command" =~ (^|[[:space:]])npm[[:space:]] ]]; then
    echo "BLOCKED: Don't use npm directly!" >&2
    echo "Use './build_frontend.sh' to build the frontend!" >&2
    exit 2
fi

# Check if trying to run dnne.bat
if [[ "$tool_name" == "Bash" ]] && [[ "$command" =~ (^|[[:space:]])([./]*)?dnne\.bat($|[[:space:]]) ]]; then
    echo "BLOCKED: DNNE runs on Windows! You are in WSL." >&2
    echo "Use MCP to restart DNNE." >&2
    exit 2
fi

# Check for timeout command with python runner.py
if [[ "$tool_name" == "Bash" ]] && [[ "$command" =~ (^|[[:space:]])timeout[[:space:]]+([0-9]+)[[:space:]]+python[[:space:]]+runner\.py ]]; then
    echo "BLOCKED: Don't use the timeout command with runner.py!" >&2
    echo "Use 'python runner.py --timeout ${BASH_REMATCH[2]}' instead." >&2
    exit 2
fi

# Check for pip install pyyaml
if [[ "$tool_name" == "Bash" ]] && [[ "$command" =~ (^|[[:space:]])pip[[:space:]]+install[[:space:]]+[^[:space:]]*pyyaml ]]; then
    echo "BLOCKED: YAML already installed, you did not activate the correct conda environment!" >&2
    echo "Use: source /home/asantanna/miniconda/bin/activate DNNE_PY38 && your_command" >&2
    exit 2
fi

exit 0
