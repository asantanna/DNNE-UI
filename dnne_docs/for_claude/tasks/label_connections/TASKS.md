# Label Connection System Tasks

*Last Updated: 2025-08-22*  
*Status: 🟢 Core Complete | 🟡 Minor Issues*

## Current Status
Label-based connection system implemented and functional. Users can create named connection points instead of direct wires, making workflows cleaner and more organized.

## Active TODOs

### High Priority
- [ ] Fix workflow naming issue - saved workflows loading as 'Unsaved Workflow (N)' instead of proper name
- [ ] Investigate tab creation logic for unsaved workflows - closing tabs sometimes creates new ones

### Low Priority  
- [ ] Add visual styling to distinguish input/output labels more clearly
- [ ] Add label search/filter in context menu when many labels exist
- [ ] Consider label grouping/categories for large workflows

## Completed Features
✅ Visual-only label nodes (no actual graph connections)  
✅ Shared label creation code with is_input parameter  
✅ Export system preprocessing via generate_label_connections()  
✅ Label connections stored in workflow_labels dictionary  
✅ Graph traversal functions aware of label connections  
✅ Proper slot configuration (inputs have outputs, outputs have inputs)  
✅ Correct positioning (right-align for inputs, left-align for outputs)  
✅ TypeScript interface updated with connectedToLabel property  
✅ Frontend builds successfully with all changes

## Quick Reference
- Label nodes are virtual (isVirtualNode = true)
- Labels stored in workflow.extra.labelDictionary
- Export resolves labels to actual connections transparently
- Individual node exporters unaware of label system