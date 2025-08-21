#!/usr/bin/env python3
"""
Toggle debug prints marked with #DBG_TAG# between commented and uncommented state.
Detects the current mode based on the first #DBG_TAG# found.
"""

import sys
import re

def toggle_dbg_tags(filename):
    """Toggle all #DBG_TAG# lines between commented and uncommented"""
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find the first #DBG_TAG# to determine current mode
    comment_mode = None
    for line in lines:
        if '#DBG_TAG#' in line:
            stripped = line.strip()
            # Check if it's a commented line (starts with # but not a shebang)
            if stripped.startswith('#') and not stripped.startswith('#!'):
                # Currently commented, will uncomment
                comment_mode = "uncomment"
            else:
                # Currently uncommented, will comment
                comment_mode = "comment"
            break
    
    if comment_mode is None:
        print(f"No #DBG_TAG# found in {filename}")
        return
    
    print(f"Mode detected: {comment_mode} (will {comment_mode} all #DBG_TAG# lines)")
    
    # Process all lines
    new_lines = []
    changes = 0
    
    for i, line in enumerate(lines):
        if '#DBG_TAG#' in line:
            if comment_mode == "uncomment":
                # Remove comment if present
                match = re.match(r'^(\s*)#\s*(.+#DBG_TAG#.*)$', line)
                if match:
                    indent = match.group(1)
                    rest = match.group(2)
                    new_line = f"{indent}{rest}\n"
                    new_lines.append(new_line)
                    changes += 1
                else:
                    # Already uncommented
                    new_lines.append(line)
            else:  # comment mode
                # Add comment if not present
                match = re.match(r'^(\s*)(.+#DBG_TAG#.*)$', line)
                if match and not line.strip().startswith('#'):
                    indent = match.group(1)
                    rest = match.group(2)
                    new_line = f"{indent}# {rest}\n"
                    new_lines.append(new_line)
                    changes += 1
                else:
                    # Already commented
                    new_lines.append(line)
        else:
            new_lines.append(line)
    
    # Write back
    with open(filename, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Toggled {changes} #DBG_TAG# lines in {filename}")
    print(f"All debug prints are now {'enabled' if comment_mode == 'uncomment' else 'disabled'}")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <filename>")
        print("Toggle debug prints marked with #DBG_TAG# between commented and uncommented state")
        sys.exit(1)
    
    filename = sys.argv[1]
    toggle_dbg_tags(filename)

if __name__ == "__main__":
    main()