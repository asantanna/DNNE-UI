#!/bin/bash
# Helper to read dnne_config.json values
# Usage: source dnne_config_reader.sh

get_dnne_config() {
    local value=$(python3 -c "import dnne_config; print(dnne_config.config.get('$1', ''))" 2>/dev/null) || {
        echo "Error: Failed to load dnne_config" >&2
        exit 1
    }
    
    # Expand ~ to home directory if present
    if [[ "$value" == ~* ]]; then
        value="${value/#\~/$HOME}"
    fi
    
    echo "$value"
}

# Export the function so it's available in scripts that source this file
export -f get_dnne_config