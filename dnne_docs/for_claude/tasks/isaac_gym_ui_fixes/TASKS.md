# IsaacGymEnvs UI Fixes - Task Tracking

## Current Issues

### 1. Hidden Dynamic Widgets Leave Gaps (Low Priority)
**Problem**: When dynamic widgets are hidden (e.g., when a task has fewer hierarchy levels), they still occupy vertical space in the node, creating visual gaps.

**Details**:
- Hidden widgets have `widget.hidden = true` set
- LiteGraph doesn't recalculate Y positions when widgets are hidden
- Results in empty space between visible widgets

**Potential Solutions**:
- Manually recalculate widget Y positions after hiding
- Use LiteGraph's widget repositioning methods if available
- Remove and re-add widgets instead of hiding

### 2. ✅ Node Doesn't Re-render on Task Change (FIXED)
**Problem**: When selecting a new task, the node size doesn't update until mouse movement triggers a redraw.

**Solution Implemented**:
```javascript
targetNode.setSize(targetNode.computeSize());
app.graph.setDirtyCanvas(true, true);
requestAnimationFrame(() => {
    app.canvas.draw(true, true);
});
```

**Fix Details**: 
- Used `setDirtyCanvas(true, true)` with both flags to mark immediate redraw needed
- Added `requestAnimationFrame` to ensure draw happens on next frame
- Called `app.canvas.draw(true, true)` within the animation frame for forced redraw
- Applied fix to all three callback handlers: task onChange, task onLoad, and dynamic widget onChange

### 3. ✅ Schema Display Widget Height Ignored (FIXED)
**Problem**: The `schema_display` STRING widget should be taller, but the height setting was being ignored.

**Solution Implemented**:
- Added `widgetHeight` property support in frontend (`useStringWidget.ts`)
- Widget now accepts `widgetHeight` parameter from backend
- Set CSS custom property `--comfy-widget-min-height` on textarea element
- Backend passes `widgetHeight: 200` for proper sizing

**Fix Details**:
- Modified frontend to check for `widgetHeight` in inputSpec and apply it via CSS custom property
- Added `widgetHeight` to zStringInputOptions schema for TypeScript support
- Removed element count text from schema display to prevent line wrapping
- Widget now properly displays at 200px height with clean formatting

## Priority
1. ~~**High**: Node re-render issue (affects usability)~~ ✅ FIXED
2. ~~**Medium**: Schema display height (affects readability)~~ ✅ FIXED
3. **Low**: Hidden widget gaps (cosmetic issue)