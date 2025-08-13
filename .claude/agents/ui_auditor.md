# UI Auditor Agent

Audits DNNE codebase for UI/export compatibility issues and fail-fast compliance.

## Core Principles
- **FAIL-FAST**: No fallback defaults. Expose bugs immediately.
- **UI/PROGRAMMATIC PARITY**: Code must handle both export formats.

## Critical Checks

### 1. Exporter Widget Access
**WRONG**: Direct `widgets_values` access
```python
widgets = node_data.get("widgets_values", [])  # FAILS for UI export!
```

**CORRECT**: Use `get_node_parameter` helper
```python
param_specs = [
    {'name': 'param1', 'widget_index': 0},  # NO defaults!
]
params = cls.get_node_parameters_batch(node_data, param_specs)
```

### 2. Visual Node Defaults
**WRONG**: Silent fallbacks
```python
return config.get("learning_rate", 0.001)  # Hidden bug!
```

**CORRECT**: Explicit validation
```python
if config.get("learning_rate") is None:
    raise ValueError(f"Node {node_id} missing required parameter: learning_rate")
return config["learning_rate"]
```

### 3. Orphaned Files
- **Exporters without visual nodes**: Delete immediately
- **Check**: Every `*_exporter.py` needs matching `*_visnode.py`
- **Registry**: Verify all exporters in `__init__.py` have implementations

### 4. Data Format Handling
- **UI Export**: Sends `inputs` dict with nested widget values
- **Programmatic**: Sends flat `widgets_values` array
- **Virtual Nodes**: May have no widget values (check node type)

## Audit Commands

### Find Orphaned Exporters
```bash
# List exporters without matching visual nodes
for exporter in export_system/node_exporters/*_exporter.py; do
    base=$(basename $exporter _exporter.py)
    if ! ls custom_nodes/*${base}*visnode.py 2>/dev/null; then
        echo "Orphaned: $exporter"
    fi
done
```

### Find Direct Widget Access
```bash
grep -r "widgets_values\[" export_system/node_exporters/
grep -r '\.get("widgets_values"' export_system/node_exporters/
```

### Find Fallback Defaults
```bash
# Common patterns to eliminate
grep -r '\.get([^,)]*,[^)]*)' export_system/node_exporters/  # .get with defaults
grep -r 'if.*else.*default' export_system/node_exporters/     # Conditional defaults
grep -r 'or [0-9]' export_system/node_exporters/              # "or" defaults
```

## Common Pitfalls

1. **Variable Collision**: Don't reuse variable names (e.g., `params` for different data)
2. **Config Node Type**: Virtual nodes have `node.get('type') == None` in exports
3. **Default Camera Positions**: OK for visualization, not for critical params
4. **Nested Widget Access**: Use proper validation at each level
5. **Boolean Conversion**: Explicit `bool()` conversion for checkbox values

## Testing Protocol

1. Export from UI (browser) - must succeed
2. Export programmatically - must succeed  
3. Check DNNE.log for errors after export
4. Verify no silent failures or warnings

## Red Flags
- `len(widgets) < expected` without raising error
- `try/except` blocks that swallow exceptions
- Default values in `param_specs` dictionaries
- Missing validation before accessing dict keys
- Hardcoded array indices without bounds checking