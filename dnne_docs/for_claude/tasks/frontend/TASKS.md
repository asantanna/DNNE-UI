# Frontend Tasks

## Current Status
UI functional with basic copy/paste support and label rats nest visualization.
Label rats nest feature complete - shows connections between labels when selected.

## Active TODOs

### Known Issues
1. **DNNE Combo Widget WebSocket Error**
   - [ ] Fix WebSocket CONNECTING state error on reload
   - Error occurs in `useDnneComboWidget.ts` line 76
   - Non-critical - doesn't affect functionality

### Low Priority
1. **Menu Enhancements**
   - [ ] Add "Copy" menu item to Edit menu (Ctrl+C)
   - [ ] Add "Paste" menu item to Edit menu (Ctrl+V)
   - [ ] Add "Paste with Links" menu item to Edit menu (Ctrl+Shift+V)

### Future Considerations
1. **Node Library**
   - [ ] Add search functionality in node sidebar
   - [ ] Implement node categories filtering
   - [ ] Add favorites/recent nodes section

2. **Canvas Improvements**
   - [ ] Add minimap for large workflows
   - [ ] Implement node alignment tools
   - [ ] Add connection rerouting helpers

## Implementation Notes

### Copy/Paste Menu Items
- Located in: `src/composables/useCoreCommands.ts`
- Copy functionality: `src/composables/useCopy.ts`
- Paste functionality: `src/composables/usePaste.ts`
- Menu structure: Edit menu in menubar