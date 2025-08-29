# Friendly Node Names in Logs

**Priority**: Low  
**Created**: 2025-08-29  
**Status**: Not Started

## Description
Update export templates to use friendly node names in logs instead of just node IDs (e.g., "MNIST Dataset (37)" instead of "node_37").

## Motivation
- Improves readability of exported code logs
- Makes debugging easier by providing context about what each node does
- Node titles are already available in workflow JSON but not utilized during export

## Implementation Notes
1. Extract node titles from workflow JSON during export (field `title` in each node)
2. Pass friendly names to node templates as an additional variable
3. Update base `QueueNode` class to accept optional friendly name parameter
4. Modify logging statements to include friendly name alongside node ID

## Example
Current log output:
```
INFO:node_37:Starting node node_37
```

Desired log output:
```
INFO:node_37:Starting MNIST Dataset (37)
```

## Files to Modify
- `/home/asantanna/DNNE/DNNE-UI/export_system/graph_exporter.py` - Extract and pass node titles
- `/home/asantanna/DNNE/DNNE-UI/export_system/templates/base/queue_framework.py` - Accept friendly name in QueueNode
- Various node templates in `/home/asantanna/DNNE/DNNE-UI/export_system/templates/nodes/` - Use friendly names in logs

## Dependencies
None - all required information already exists in workflow JSON

## Estimated Effort
2-3 hours