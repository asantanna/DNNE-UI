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

### 3. Schema Display Widget Height Ignored
**Problem**: The `schema_display` STRING widget should be 3x taller, but the height setting is being ignored.

**Current Setting**:
```python
"schema_display": ("STRING", {
    "multiline": True,
    "default": "",
    "tooltip": "Current observation and action schema",
    "readonly": True,
    "height": 600,  # This is being ignored
})
```

**Issue**: DOM-based widgets (STRING with multiline) might not respect the height parameter

**Potential Solutions**:
- Set height via CSS after widget creation
- Use JavaScript to modify the textarea element directly
- Override widget creation to set custom height
- Check if there's a different parameter name for DOM widget height

## Priority
1. ~~**High**: Node re-render issue (affects usability)~~ ✅ FIXED
2. **Medium**: Schema display height (affects readability)
3. **Low**: Hidden widget gaps (cosmetic issue)