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

# Check if it's trying to run graph_exporter.py
if [[ "$tool_name" == "Bash" ]] && [[ "$command" =~ (^|[[:space:]])python[0-9]*[[:space:]]+.*graph_exporter\.py($|[[:space:]]) ]]; then
    echo "BLOCKED: Don't use graph_exporter.py - it's deprecated!" >&2
    echo "Use claude_scripts/programmatic_export.py instead!" >&2
    exit 2
fi

# Check for stderr redirection followed by pipe
if [[ "$tool_name" == "Bash" ]] && [[ "$command" =~ (2>&1|&>|>&)[[:space:]]*\| ]]; then
    echo "BLOCKED: Do not use stderr redirection when using pipes. This does not work!" >&2
    echo "Instead, use parentheses: (command 2>&1) | next_command" >&2
    exit 2
fi

# Block ALL direct npm usage
if [[ "$tool_name" == "Bash" ]] && [[ "$command" =~ (^|[[:space:]])npm[[:space:]] ]]; then
    echo "BLOCKED: Don't use npm directly!" >&2
    echo "Use './build_frontend.sh' to build the frontend!" >&2
    exit 2
fi

exit 0
