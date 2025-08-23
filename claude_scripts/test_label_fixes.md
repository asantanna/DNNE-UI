# Label System Bug Fixes Test Plan

## Fixed Issues

### 1. Label text not drawing when loading saved workflows
**Fix Applied**: Added `onConfigure()` method to restore label properties from serialized data, and storing labelName/labelDirection in `this.properties` for serialization.

**Test Steps**:
1. Create a workflow with labels (both input and output)
2. Save the workflow
3. Close/reload the UI
4. Load the saved workflow
5. **Expected**: Labels should display their text correctly

### 2. Context menu not showing "Connect to Label >" after deleting input label
**Fix Applied**: 
- Added `dictionaryKey` property to track the actual dictionary key
- Modified `removeLabelFromDictionary()` to use the correct key
- Added check in context menu to prevent showing menu if input already has a label

**Test Steps**:
1. Create an output label (e.g., from a Tensor node output)
2. Connect an input to that label (creates an input label)
3. Delete the input label
4. Try to drag from the same input again
5. **Expected**: "Connect to Label >" menu should appear again

## Implementation Details

### Dictionary Key Storage
- Output labels: key = labelName
- Input labels: key = `${nodeType}(${nodeId}).input_${slotIndex}`

### Serialization
- All label properties (labelName, labelDirection, dictionaryKey) now stored in `this.properties` 
- `onConfigure()` restores these from saved workflow data

### Context Menu Logic
- Now checks if `${node.constructor.type || node.type}(${node.id}).input_${slot}` exists in dictionary
- Only shows "Connect to Label >" if no existing label for that input slot

## Files Modified
- `/home/asantanna/DNNE/DNNE-UI-Frontend/src/extensions/core/labelNode.ts`

## Next Steps
Test both scenarios in the DNNE UI to confirm the fixes work as expected.