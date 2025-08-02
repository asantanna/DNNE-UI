#!/bin/bash

# Read the JSON input from Claude Code
json_input=$(cat)
tool_name=$(echo "$json_input" | jq -r '.tool_name')
command=$(echo "$json_input" | jq -r '.tool_input.command // empty')

# Check if it's trying to run graph_exporter.py
if [[ "$tool_name" == "Bash" ]] && [[ "$command" =~ python.*graph_exporter\.py ]]; then
    echo "BLOCKED: Don't use graph_exporter.py - it's deprecated!" >&2
    echo "Use claude_scripts/programmatic_export.py instead!" >&2
    exit 2
fi

exit 0