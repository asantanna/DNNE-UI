#!/usr/bin/env python3
"""
Check which exporters are missing schema methods
"""
from pathlib import Path

exporters_dir = Path("/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/node_exporters")

# List of critical schema methods
schema_methods = ["get_output_schema", "get_initial_output_schema"]

missing = {}

for filepath in sorted(exporters_dir.glob("*_exporter.py")):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Skip virtual nodes
    if "is_virtual" in content and "return True" in content:
        continue
        
    missing_methods = []
    for method in schema_methods:
        if f"def {method}" not in content:
            missing_methods.append(method)
    
    if missing_methods:
        missing[filepath.name] = missing_methods

print("Exporters missing schema methods:")
for file, methods in missing.items():
    print(f"\n{file}:")
    for method in methods:
        print(f"  - {method}")

print(f"\nTotal: {len(missing)} exporters need schema methods")