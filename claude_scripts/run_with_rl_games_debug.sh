#!/bin/bash
# Run IsaacGymEnvs with our debug version of rl_games

# Rename rl_games_debug to rl_games temporarily
if [ -d "$HOME/DNNE-LINUX-SUPPORT/rl_games_debug" ] && [ ! -d "$HOME/DNNE-LINUX-SUPPORT/rl_games" ]; then
    mv "$HOME/DNNE-LINUX-SUPPORT/rl_games_debug" "$HOME/DNNE-LINUX-SUPPORT/rl_games"
    echo "Using debug version of rl_games"
fi

# Add to PYTHONPATH to prioritize our version
export PYTHONPATH="$HOME/DNNE-LINUX-SUPPORT:$PYTHONPATH"

# Run the command with all arguments passed through
"$@"

# Rename back to avoid confusion
if [ -d "$HOME/DNNE-LINUX-SUPPORT/rl_games" ] && [ ! -d "$HOME/DNNE-LINUX-SUPPORT/rl_games_debug" ]; then
    mv "$HOME/DNNE-LINUX-SUPPORT/rl_games" "$HOME/DNNE-LINUX-SUPPORT/rl_games_debug"
fi