# Label Rats Nest Visualization

## Overview
The Label Rats Nest feature provides visual feedback when working with label connections in DNNE. When a Label node is selected, the system displays thin, straight lines connecting all labels in the same network, similar to "rats nest" visualization in electronics CAD software.

## Features
- **Automatic Network Detection**: Selecting any Label node instantly shows all connected labels with the same name
- **Visual Distinction**: Uses cyan-colored thin lines to differentiate from normal node connections
- **Straight Line Connections**: Draws direct lines between label centers (not curved paths)
- **Animated Highlights**: Selected label network pulses with a subtle glow effect
- **Performance Optimized**: Caches label networks to avoid recalculation

## Visual Design
- **Line Color**: Cyan (rgba(0, 255, 255, 0.5)) with 50% opacity
- **Line Width**: 1px for minimal visual intrusion
- **Shadow Effect**: Subtle cyan glow for better visibility
- **Label Highlight**: Dashed border with pulsing intensity around connected labels

## Usage
1. Click on any Label node to select it
2. All labels in the same network (same labelName) are immediately highlighted
3. Straight lines connect the output label to all input labels
4. Click elsewhere to clear the visualization

## Implementation Details
- **File**: `src/extensions/core/dnne_labelRatsNest.ts`
- **Extension Name**: `DNNE.LabelRatsNest`
- **Hooks**: Canvas drawing (drawFrontCanvas) and selection monitoring

## Console Commands
```javascript
// Toggle rats nest on/off
window.dnneRatsNestManager.toggle()

// Disable rats nest
window.dnneRatsNestManager.toggle(false)

// Enable rats nest
window.dnneRatsNestManager.toggle(true)
```

## Performance
- Label networks are cached per label name
- Cache invalidates when nodes are added/removed
- Animation uses requestAnimationFrame for smooth updates
- Minimal overhead during normal canvas operations