#!/usr/bin/env python3
"""
Setup script to use rl_games_debug instead of the installed rl_games.
This allows us to add debug logging without modifying the site-packages version.
"""

import sys
import os

# Add rl_games_debug to the beginning of sys.path
rl_games_debug_path = os.path.expanduser("~/DNNE-LINUX-SUPPORT")
if rl_games_debug_path not in sys.path:
    sys.path.insert(0, rl_games_debug_path)
    print(f"Added {rl_games_debug_path} to sys.path")

# Now imports of rl_games will use rl_games_debug
# Rename the folder to match the import name
debug_path = os.path.join(rl_games_debug_path, "rl_games_debug")
rl_games_path = os.path.join(rl_games_debug_path, "rl_games")

if os.path.exists(debug_path) and not os.path.exists(rl_games_path):
    os.rename(debug_path, rl_games_path)
    print(f"Renamed rl_games_debug to rl_games for import compatibility")

print("rl_games_debug setup complete!")
print("Now 'import rl_games' will use the debug version.")