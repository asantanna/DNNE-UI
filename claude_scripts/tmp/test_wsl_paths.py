#!/usr/bin/env python3
"""
Test script to diagnose WSL path access from Windows
Run this from Windows to check if WSL paths are accessible
"""

import os
import sys
import platform
from pathlib import Path

def test_path_access(path_str, description):
    """Test if a path is accessible"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Path: {path_str}")
    print(f"-"*60)
    
    try:
        path = Path(path_str)
        
        # Test existence
        exists = path.exists()
        print(f"  exists(): {exists}")
        
        if exists:
            # Test if it's a directory
            is_dir = path.is_dir()
            print(f"  is_dir(): {is_dir}")
            
            # Test if it's a file
            is_file = path.is_file()
            print(f"  is_file(): {is_file}")
            
            # Try to list contents if directory
            if is_dir:
                try:
                    contents = list(path.iterdir())
                    print(f"  Can list contents: Yes ({len(contents)} items)")
                    # Show first few items
                    for item in contents[:3]:
                        print(f"    - {item.name}")
                    if len(contents) > 3:
                        print(f"    ... and {len(contents)-3} more")
                except Exception as e:
                    print(f"  Can list contents: No - {e}")
            
            # Try to read if it's a file
            if is_file:
                try:
                    with open(path, 'r') as f:
                        first_line = f.readline()
                    print(f"  Can read file: Yes")
                    print(f"  First line: {first_line[:50]}...")
                except Exception as e:
                    print(f"  Can read file: No - {e}")
        
        return True
        
    except OSError as e:
        print(f"  ERROR (OSError): {e}")
        print(f"  Error code: {e.errno if hasattr(e, 'errno') else 'N/A'}")
        if hasattr(e, 'winerror'):
            print(f"  Windows error: {e.winerror}")
        return False
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return False

def main():
    print("="*60)
    print("WSL Path Access Diagnostic Tool")
    print("="*60)
    print(f"Platform: {platform.system()}")
    print(f"Python: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    # Test various WSL path formats
    test_paths = [
        # UNC path format (what Windows typically uses)
        (r"\\wsl.localhost\Ubuntu\home\asantanna", "WSL UNC path (home directory)"),
        (r"\\wsl.localhost\Ubuntu\home\asantanna\DNNE", "WSL UNC path (DNNE directory)"),
        (r"\\wsl.localhost\Ubuntu\home\asantanna\DNNE\DNNE-LINUX-SUPPORT", "WSL UNC path (Linux support)"),
        (r"\\wsl.localhost\Ubuntu\home\asantanna\DNNE\DNNE-LINUX-SUPPORT\IsaacGymEnvs", "WSL UNC path (IsaacGymEnvs)"),
        (r"\\wsl.localhost\Ubuntu\home\asantanna\DNNE\DNNE-LINUX-SUPPORT\IsaacGymEnvs\isaacgymenvs\cfg\config.yaml", "WSL UNC path (config.yaml)"),
        
        # Alternative UNC format
        (r"\\wsl$\Ubuntu\home\asantanna", "WSL$ UNC path (home directory)"),
        (r"\\wsl$\Ubuntu\home\asantanna\DNNE\DNNE-LINUX-SUPPORT\IsaacGymEnvs\isaacgymenvs\cfg\config.yaml", "WSL$ UNC path (config.yaml)"),
        
        # U: drive format (if WSL is mounted as U:)
        (r"U:\home\asantanna", "U: drive (home directory)"),
        (r"U:\home\asantanna\DNNE\DNNE-LINUX-SUPPORT", "U: drive (Linux support)"),
        (r"U:\home\asantanna\DNNE\DNNE-LINUX-SUPPORT\IsaacGymEnvs", "U: drive (IsaacGymEnvs)"),
        (r"U:\home\asantanna\DNNE\DNNE-LINUX-SUPPORT\IsaacGymEnvs\isaacgymenvs\cfg\config.yaml", "U: drive (config.yaml)"),
    ]
    
    results = []
    for path, description in test_paths:
        success = test_path_access(path, description)
        results.append((path, description, success))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    working_paths = [r for r in results if r[2]]
    failed_paths = [r for r in results if not r[2]]
    
    if working_paths:
        print(f"\n✅ Working paths ({len(working_paths)}):")
        for path, desc, _ in working_paths:
            print(f"  - {desc}")
    
    if failed_paths:
        print(f"\n❌ Failed paths ({len(failed_paths)}):")
        for path, desc, _ in failed_paths:
            print(f"  - {desc}")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if working_paths:
        # Find the best working format
        if any("U:" in r[0] for r in working_paths):
            print("✅ U: drive format is working - use this for paths")
            print("   Example: U:\\home\\asantanna\\DNNE\\DNNE-LINUX-SUPPORT")
        elif any(r"\\wsl.localhost" in r[0] for r in working_paths):
            print("✅ \\\\wsl.localhost format is working")
            print("   Example: \\\\wsl.localhost\\Ubuntu\\home\\asantanna")
        elif any(r"\\wsl$" in r[0] for r in working_paths):
            print("✅ \\\\wsl$ format is working")
            print("   Example: \\\\wsl$\\Ubuntu\\home\\asantanna")
    else:
        print("❌ No WSL paths are accessible from Windows")
        print("   Possible causes:")
        print("   1. WSL is not running (try: wsl --list --running)")
        print("   2. WSL network service is down")
        print("   3. Windows Defender or firewall blocking access")
        print("   4. WSL instance name might be different (not 'Ubuntu')")
        print("\n   Try running: wsl --list --verbose")
        print("   to see your WSL instances and their states")

if __name__ == "__main__":
    main()