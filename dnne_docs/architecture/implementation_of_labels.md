# Implementation of Labels in DNNE

## Overview
DNNE labels provide a way to create "wireless" connections between nodes, similar to goto/label pairs in programming. All connection information is stored directly in the Label node properties, making the system simple and reliable.

## Architecture

### Label Node Properties

**Output Labels** store:
- `labelName`: Display text (e.g., "LinearLayer(46).output")
- `labelDirection`: "output"
- `sourceNodeId`: ID of the node providing data
- `sourceSlotIndex`: Slot index on source node
- `sourceSlotName`: Slot name for validation
- `sourceSlotType`: Type of the slot

**Input Labels** store:
- `labelName`: Display text (matches output label)
- `labelDirection`: "input"
- `targetNodeId`: ID of the node receiving data
- `targetSlotIndex`: Slot index on target node
- `targetSlotName`: Slot name for validation
- `targetSlotType`: Type of the slot
- `connectedToLabel`: Name of the output label to connect to

### Resolution Process

During export, the system:
1. Scans all Label nodes in the workflow
2. Collects output labels by name
3. Matches input labels to output labels via `connectedToLabel`
4. Creates direct connections between source and target nodes
5. Filters out Label nodes from the exported code

### Validation

The export system validates labels and provides clear error messages for:
- **Orphaned Output Labels**: Missing source connection info
- **Orphaned Input Labels**: Missing target connection info
- **Missing Output Labels**: Input labels referencing non-existent outputs
- **Duplicate Output Labels**: Multiple outputs with the same name

## Key Benefits

1. **Single Source of Truth**: Node properties contain all information
2. **Simple and Reliable**: No complex synchronization needed
3. **Predictable Behavior**: No automatic cleanup that might surprise users
4. **Clear Error Messages**: Export-time validation with actionable feedback
5. **Future-Proof**: Label text can be made editable without breaking connections

## UI Behavior

The UI allows all label operations without automatic cleanup:
- Deleting nodes doesn't auto-delete connected labels
- Deleting labels doesn't affect other labels
- Users manually manage label lifecycle

This makes the system more predictable and easier to debug. Any issues are caught at export time with clear instructions on how to fix them.

## Example

```json
{
  "id": 20,
  "type": "Label",
  "properties": {
    "labelName": "TensorConstant(10).output",
    "labelDirection": "output",
    "sourceNodeId": 10,
    "sourceSlotIndex": 0,
    "sourceSlotName": "output",
    "sourceSlotType": "TENSOR"
  }
}
```

This output label can be referenced by multiple input labels using the `connectedToLabel` property.