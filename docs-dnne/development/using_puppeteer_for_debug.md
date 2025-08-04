# Using Puppeteer for DNNE UI Debugging

This document provides a comprehensive reference for interacting with the DNNE UI using Puppeteer. It includes all important selectors, configuration details, and helper functions for automated UI testing and debugging.

## Table of Contents
- [Configuration](#configuration)
- [UI Element Selectors](#ui-element-selectors)
  - [Sidebar Tabs](#sidebar-tabs)
  - [Top Menu Bar](#top-menu-bar)
  - [Action Bar](#action-bar)
  - [Graph Canvas Controls](#graph-canvas-controls)
  - [Workflow Management](#workflow-management)
  - [Node Library](#node-library)
  - [Queue Tab](#queue-tab)
  - [Dialogs and Modals](#dialogs-and-modals)
- [Menu Commands](#menu-commands)
- [Helper Functions](#helper-functions)
- [Common Tasks](#common-tasks)
- [Known Issues](#known-issues)

## Configuration

### Puppeteer Launch Configuration

**Use this exact configuration to properly display the DNNE UI:**

```javascript
await mcp__puppeteer__puppeteer_navigate({
  url: "http://172.22.160.1:8188",
  launchOptions: {
    "headless": false, 
    "defaultViewport": null, 
    "args": ["--start-maximized"]
  }
});
```

### Important Configuration Notes
- **URL**: Access DNNE from WSL2 at `http://172.22.160.1:8188`
- **Must use `--start-maximized`** to see the status bar
- **Do NOT use zoom** (e.g., `document.body.style.zoom`) as it breaks the maximized state
- Status bar shows: "Agent: ⚪ Connected | Clients: 0" on left, Export/Run controls on right
- Tested on 1920x1080 displays

## UI Element Selectors

### Sidebar Tabs

The sidebar contains tabs for different functionality. Each tab has a unique class selector:

| Tab | Selector | Description |
|-----|----------|-------------|
| Workflows | `.workflows-tab-button` | Manage and organize workflows |
| Node Library | `.node-library-tab-button` | Browse available nodes |
| Model Library | `.model-library-tab-button` | Manage ML models |
| Queue | `.queue-tab-button` | View and manage execution queue |

**Generic Classes:**
- `.side-bar-button` - All sidebar buttons
- `.side-bar-button-icon` - Icons within sidebar buttons
- `.side-tool-bar-container` - Main sidebar container

### Top Menu Bar

| Element | Selector | Description |
|---------|----------|-------------|
| Menu Container | `.comfyui-menu` | Main menu bar container |
| Menu Bar | `.top-menubar` | PrimeVue menubar component |
| Menu Items | `.p-menubar-item-link` | Individual menu item links |
| Menu Labels | `.p-menubar-item-label` | Menu item text labels |
| Workflow Menu | `.p-menubar-root-list > li:nth-child(1)` | First menu (Workflow) |
| Edit Menu | `.p-menubar-root-list > li:nth-child(2)` | Second menu (Edit) |
| DNNE Menu | `.p-menubar-root-list > li:nth-child(3)` | Third menu (DNNE) |
| Hide Menu | `[aria-label="Hide menu"]` | Toggle focus mode |
| Logo | `.comfyui-logo` | DNNE/ComfyUI logo |

### Action Bar

The action bar contains the main workflow execution controls:

| Element | Selector | Description |
|---------|----------|-------------|
| Export Button | `[data-testid="export-button"]` | Main export button |
| Export Target Dropdown | `.export-target-dropdown` | Select export destination |
| Run After Export | (checkbox element) | Auto-run after export |

### Graph Canvas Controls

Located in the bottom-right corner of the canvas:

| Control | Selector | Description |
|---------|----------|-------------|
| Canvas Menu | `.p-buttongroup-vertical` | Button group container |
| Zoom In | `[aria-label="Zoom In"]` | Zoom in button |
| Zoom Out | `[aria-label="Zoom Out"]` | Zoom out button |
| Fit View | `[aria-label="Fit View"]` | Fit graph to view |
| Toggle Links | `[data-testid="toggle-link-visibility-button"]` | Show/hide connection lines |

### Workflow Management

| Element | Selector | Description |
|---------|----------|-------------|
| Workflow Tabs | `.workflow-tabs` | Tab container for open workflows |
| Close Workflow | `.close-workflow-button` | Close individual workflow |
| New Workflow | (via menu command) | Create new blank workflow |
| Workflow Tree Items | `li[aria-label="filename"] .p-tree-node-content` | Click on workflow items in tree |

**Important:** To click on workflow items in the tree explorer, use the `aria-label` selector:
```javascript
// Example: Loading MNIST_Test.json workflow
await mcp__puppeteer__puppeteer_click({
  selector: 'li[aria-label="MNIST_Test.json"] .p-tree-node-content'
});
```

### Node Library

| Control | Selector | Description |
|---------|----------|-------------|
| New Folder | `.new-folder-button` | Create new folder |
| Sort | `.sort-button` | Sort nodes |
| Filter | `.filter-button` | Filter node list |
| Tree Explorer | `.tree-explorer` | Node tree navigation |

### Queue Tab

| Element | Selector | Description |
|---------|----------|-------------|
| Task Item | `.task-item` | Individual queue items |
| Queue Controls | (context menu) | Right-click for options |

### Dialogs and Modals

Common dialog elements use PrimeVue components:
- `.p-dialog` - Dialog container
- `.p-dialog-header` - Dialog header
- `.p-dialog-content` - Dialog content area
- `.p-dialog-footer` - Dialog footer with buttons

## Menu Commands

The top menu bar provides access to these commands:

### Opening Menus and Clicking Items

To interact with menu items, use a two-step process:
1. Click the menu header to open the dropdown
2. Click the specific menu item in the submenu

```javascript
// Open Workflow menu
await mcp__puppeteer__puppeteer_click({
  selector: ".p-menubar-root-list > li:nth-child(1) .p-menubar-item-label"
});

// Click first item in submenu (e.g., "New")
await mcp__puppeteer__puppeteer_click({
  selector: ".p-menubar-submenu li:first-child .p-menubar-item-link"
});

// Click second item in submenu (e.g., "Open")
await mcp__puppeteer__puppeteer_click({
  selector: ".p-menubar-submenu li:nth-child(2) .p-menubar-item-link"
});
```

### Workflow Menu
- **New Blank Workflow** - `Comfy.NewBlankWorkflow` (1st item)
- **Open Workflow** - `Comfy.OpenWorkflow` (2nd item)
- **Browse Templates** - `Comfy.BrowseTemplates` (3rd item)
- **Save Workflow** - `Comfy.SaveWorkflow` (4th item)
- **Save Workflow As** - `Comfy.SaveWorkflowAs` (5th item)
- **Export Workflow** - `Comfy.ExportWorkflow` (6th item)
- **Export Workflow (API)** - `Comfy.ExportWorkflowAPI` (7th item)

### Edit Menu
- **Undo** - `Comfy.Undo`
- **Redo** - `Comfy.Redo`
- **Clear Workflow** - `Comfy.ClearWorkflow`
- **Refresh Node Definitions** - `Comfy.RefreshNodeDefinitions`
- **Open Clipspace** - `Comfy.OpenClipspace`

### DNNE Menu
- (Placeholder for future DNNE-specific features)

## Helper Functions

### Basic Navigation and Interaction

```javascript
// Initialize Puppeteer and navigate to DNNE
await mcp__puppeteer__puppeteer_navigate({
  url: "http://172.22.160.1:8188",
  launchOptions: {
    "headless": false,
    "defaultViewport": null,
    "args": ["--start-maximized"]
  }
});

// Take a screenshot
await mcp__puppeteer__puppeteer_screenshot({
  name: "dnne-ui-state",
  encoded: false
});

// Click on elements
await mcp__puppeteer__puppeteer_click({
  selector: ".workflows-tab-button"
});
```

### Sidebar Navigation

```javascript
// Open Workflows sidebar
await mcp__puppeteer__puppeteer_click({
  selector: ".workflows-tab-button"
});

// Open Node Library
await mcp__puppeteer__puppeteer_click({
  selector: ".node-library-tab-button"
});

// Open Model Library
await mcp__puppeteer__puppeteer_click({
  selector: ".model-library-tab-button"
});

// Open Queue
await mcp__puppeteer__puppeteer_click({
  selector: ".queue-tab-button"
});
```

### Graph Canvas Operations

```javascript
// Zoom controls
await mcp__puppeteer__puppeteer_click({
  selector: "[aria-label='Zoom In']"
});

await mcp__puppeteer__puppeteer_click({
  selector: "[aria-label='Zoom Out']"
});

// Fit graph to view
await mcp__puppeteer__puppeteer_click({
  selector: "[aria-label='Fit View']"
});

// Toggle connection lines visibility
await mcp__puppeteer__puppeteer_click({
  selector: "[data-testid='toggle-link-visibility-button']"
});
```

### Workflow Export

```javascript
// Click the Export button
await mcp__puppeteer__puppeteer_click({
  selector: "[data-testid='export-button']"
});

// The export will trigger based on the selected target
```

## JavaScript Evaluation

JavaScript evaluation allows you to directly interact with the page and access internal application state. Here are useful examples:

### Getting UI State Information

```javascript
await mcp__puppeteer__puppeteer_evaluate({
  script: `(() => {
    return {
      // Window dimensions
      windowWidth: window.innerWidth,
      windowHeight: window.innerHeight,
      
      // Current workflow name
      currentWorkflow: document.title,
      
      // Count of open workflow tabs
      workflowTabCount: document.querySelectorAll('[role="tab"]').length,
      
      // Check if sidebar is open
      sidebarOpen: document.querySelector('.sidebar-content-container') !== null
    };
  })();`
});
```

### Accessing Canvas State

```javascript
await mcp__puppeteer__puppeteer_evaluate({
  script: `(() => {
    if (window.app && window.app.canvas && window.app.canvas.ds) {
      return {
        scale: window.app.canvas.ds.scale,
        offset: window.app.canvas.ds.offset,
        // You can also access nodes, links, etc.
        nodeCount: window.app.canvas.graph ? window.app.canvas.graph.nodes.length : 0
      };
    }
    return 'Canvas not accessible';
  })();`
});
```

### Getting Sidebar Tab Information

```javascript
await mcp__puppeteer__puppeteer_evaluate({
  script: `(() => {
    const sidebarButtons = Array.from(document.querySelectorAll('.side-bar-button'));
    return sidebarButtons.map(btn => {
      const classes = Array.from(btn.classList);
      const tabClass = classes.find(c => c.endsWith('-tab-button'));
      return tabClass ? tabClass.replace('-tab-button', '') : 'unknown';
    });
  })();`
});
```

### Checking Element Visibility

```javascript
await mcp__puppeteer__puppeteer_evaluate({
  script: `(() => {
    const exportButton = document.querySelector('[data-testid="export-button"]');
    if (exportButton) {
      const rect = exportButton.getBoundingClientRect();
      const style = window.getComputedStyle(exportButton);
      return {
        found: true,
        visible: style.display !== 'none' && style.visibility !== 'hidden',
        position: { x: rect.x, y: rect.y },
        size: { width: rect.width, height: rect.height }
      };
    }
    return { found: false };
  })();`
});
```

### Triggering Custom Events

```javascript
await mcp__puppeteer__puppeteer_evaluate({
  script: `(() => {
    // Dispatch keyboard events
    document.dispatchEvent(new KeyboardEvent('keydown', {
      key: 'n',
      ctrlKey: true,
      bubbles: true
    }));
    
    // Trigger click on element by searching text
    const elements = Array.from(document.querySelectorAll('*'));
    const target = elements.find(el => el.textContent === 'MNIST_Test.json');
    if (target) {
      target.click();
      return 'Clicked';
    }
    return 'Not found';
  })();`
});
```

### Getting Error Messages

```javascript
await mcp__puppeteer__puppeteer_evaluate({
  script: `(() => {
    // Check for error dialogs
    const errorDialog = document.querySelector('.p-dialog');
    if (errorDialog) {
      const title = errorDialog.querySelector('.p-dialog-title')?.textContent;
      const message = errorDialog.querySelector('.p-dialog-content')?.textContent;
      return { hasError: true, title, message };
    }
    return { hasError: false };
  })();`
});
```

## Common Tasks

### 1. Opening a Workflow from the Sidebar

```javascript
// Open workflows sidebar
await mcp__puppeteer__puppeteer_click({
  selector: ".workflows-tab-button"
});

// Click on a specific workflow using aria-label
await mcp__puppeteer__puppeteer_click({
  selector: 'li[aria-label="MNIST_Test.json"] .p-tree-node-content'
});

// The workflow will load into the canvas
```

### 2. Creating a New Workflow

```javascript
// Method 1: Using keyboard shortcut (if configured)
await mcp__puppeteer__puppeteer_evaluate({
  script: "document.dispatchEvent(new KeyboardEvent('keydown', {key: 'n', ctrlKey: true}))"
});

// Method 2: Via menu (would require menu navigation)
```

### 3. Exporting the Current Workflow

```javascript
// Click the export button
await mcp__puppeteer__puppeteer_click({
  selector: "[data-testid='export-button']"
});

// The workflow will be exported to the configured destination
```

### 4. Taking a Full UI Screenshot

```javascript
// Ensure window is maximized first
await mcp__puppeteer__puppeteer_screenshot({
  name: "full-dnne-ui",
  encoded: false,
  width: 1920,
  height: 1080
});
```

## Known Issues

### Status Bar Clipping
- **Issue**: Taking screenshots can cause the status bar to be clipped
- **Cause**: Unknown - appears to be related to viewport changes during screenshot capture
- **Workaround**: Ensure window is maximized before taking screenshots

### Window Maximization
- **Issue**: Status bar not visible unless window is maximized
- **Solution**: Always use `--start-maximized` in launch options
- **Do NOT use zoom**: Using `document.body.style.zoom` will cause the window to lose its maximized state

### Multi-Monitor Setup
- **Issue**: Browser may open on the wrong monitor
- **Note**: The browser may open on monitor 2 but not maximized, or maximized on the wrong monitor
- **Workaround**: `--start-maximized` generally works best even if on wrong monitor

## Tips for Debugging

1. **Always take a screenshot first** to see the current state of the UI
2. **Use encoded screenshots** (`encoded: true`) when you need to analyze the image content programmatically
3. **Add delays** between actions to allow for UI updates and animations
4. **Check element visibility** before clicking - some elements may be hidden until certain conditions are met
5. **Use the browser console** via `puppeteer_evaluate` to debug JavaScript or check element states

## Updating This Document

When new UI elements are added or selectors change:
1. Find the selector using browser developer tools
2. Test the selector with Puppeteer
3. Add it to the appropriate section in this document
4. Include example usage if it's a complex interaction