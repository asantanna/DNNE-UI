# DNNE UI Selectors and Navigation Reference

This document contains all UI selectors, menu navigation patterns, and interaction strategies for the DNNE UI MCP Server.

## Menu Navigation

### Menu Structure
The top menu bar uses PrimeVue components with a specific navigation pattern:

```python
# Menu bar structure
MENU_BAR = ".top-menubar"
MENU_ITEMS = ".p-menubar-item-link"
MENU_LABELS = ".p-menubar-item-label"
MENU_SUBMENU = ".p-menubar-submenu"

# Specific menus (by position)
def get_menu_item_selector(index: int) -> str:
    """Get selector for menu item by index"""
    return f".p-menubar-root-list > li:nth-child({index})"

# Submenu items
def get_submenu_item_selector(index: int) -> str:
    """Get selector for submenu item by index"""
    return f".p-menubar-submenu li:nth-child({index}) .p-menubar-item-link"
```

### Menu Items Order

#### Workflow Menu (1st menu)
1. **New Blank Workflow** - 1st item
2. **(separator)** - 2nd item (empty)
3. **Open Workflow** - 3rd item
4. **Browse Templates** - 4th item (non-functional)
5. **(separator)** - 5th item (empty)
6. **Save Workflow** - 6th item
7. **Save Workflow As** - 7th item ← CORRECTED INDEX
8. **Export Workflow** - 8th item (redundant)
9. **Export Workflow (API)** - 9th item (non-functional)

#### Edit Menu (2nd menu)
1. **Undo** - 1st item
2. **Redo** - 2nd item
3. **Clear Workflow** - 3rd item
4. **Refresh Node Definitions** - 4th item
5. **Open Clipspace** - 5th item

#### DNNE Menu (3rd menu)
- Placeholder for future DNNE-specific features

### Menu Navigation Pattern

**IMPORTANT**: Menus require a two-step process:
1. Check if submenu is already visible
2. If not visible, click menu header to open
3. Click the specific item in submenu

```python
# Correct pattern (checks if menu is already open)
submenu_visible = await browser.is_visible(MENU_SUBMENU)
if not submenu_visible:
    menu_selector = get_menu_item_selector(1)  # Workflow menu
    await browser.click(f"{menu_selector} .p-menubar-item-label")
    await asyncio.sleep(0.5)

# Now click the submenu item
save_as_selector = get_submenu_item_selector(5)  # Save As
await browser.click(save_as_selector)
```

## Sidebar Selectors

### Tab Buttons
```python
WORKFLOWS_TAB = ".workflows-tab-button"
NODE_LIBRARY_TAB = ".node-library-tab-button"
MODEL_LIBRARY_TAB = ".model-library-tab-button"  # To be removed
QUEUE_TAB = ".queue-tab-button"  # To be removed
```

### Workflow Tree Navigation
```python
def get_workflow_selector(name: str) -> str:
    """Get selector for a specific workflow in the tree
    
    The workflows use aria-label attributes on li elements.
    Example: li[aria-label="MNIST_Test.json"]
    """
    return f'li[aria-label="{name}"] > .p-tree-node-content'
```

**DOM Structure** (from actual HTML):
```html
<li class="p-tree-node" aria-label="MNIST_Test.json">
    <div class="p-tree-node-content p-tree-node-selectable">
        <span class="p-tree-node-label">
            <span>MNIST_Test.json</span>
        </span>
    </div>
</li>
```

## Action Bar Selectors

```python
EXPORT_BUTTON = '[data-testid="export-button"]'
EXPORT_TARGET_DROPDOWN = ".export-target-dropdown"
RUN_AFTER_EXPORT = "#run-after-export"  # TBD - need to verify
```

## Canvas Control Selectors

Located in bottom-right corner:
```python
ZOOM_IN = '[aria-label="Zoom In"]'
ZOOM_OUT = '[aria-label="Zoom Out"]'
FIT_VIEW = '[aria-label="Fit View"]'
TOGGLE_LINKS = '[data-testid="toggle-link-visibility-button"]'
```

## Dialog Selectors

PrimeVue dialog components:
```python
DIALOG = ".p-dialog"
DIALOG_HEADER = ".p-dialog-header"
DIALOG_CONTENT = ".p-dialog-content"
DIALOG_FOOTER = ".p-dialog-footer"
DIALOG_CLOSE = ".p-dialog-header-close"
```

### Save Dialog Pattern
```python
# Wait for save dialog
dialog_visible = await browser.wait_for_selector(DIALOG, timeout=3000)

# Find input field and enter name
input_selector = f"{DIALOG} input[type='text']"
await browser.type_text(input_selector, name)

# Click Save button
save_button = f"{DIALOG_FOOTER} button:has-text('Save')"
await browser.click(save_button)
```

## Status Bar Selectors

```python
STATUS_BAR = ".status-bar"
AGENT_STATUS = ".agent-status-bar"  # Fixed from .agent-status
```

## Client/Log Selectors

```python
CLIENT_DROPDOWN = ".client-dropdown"  # TBD
LOG_PANEL = ".log-panel"  # TBD
SHOW_ALL_LOGS = ".show-all-logs"  # TBD
CLEAR_LOGS = ".clear-logs"  # TBD
```

## JavaScript Evaluation Patterns

### Check Element Visibility
```javascript
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
```

### Get Canvas State
```javascript
if (window.app && window.app.canvas && window.app.canvas.ds) {
    return {
        scale: window.app.canvas.ds.scale,
        offset: window.app.canvas.ds.offset,
        nodeCount: window.app.canvas.graph ? window.app.canvas.graph.nodes.length : 0
    };
}
```

### Check for Error Dialogs
```javascript
const errorDialog = document.querySelector('.p-dialog');
if (errorDialog) {
    const title = errorDialog.querySelector('.p-dialog-title')?.textContent;
    const message = errorDialog.querySelector('.p-dialog-content')?.textContent;
    return { hasError: true, title, message };
}
```

### Trigger Keyboard Shortcuts
```javascript
// Save workflow (Ctrl+S)
document.dispatchEvent(new KeyboardEvent('keydown', {
    key: 's',
    ctrlKey: true,
    bubbles: true
}));

// New workflow (Ctrl+N)
document.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'n',
    ctrlKey: true,
    bubbles: true
}));
```

## Known Issues and Workarounds

### Status Bar Visibility
- **Issue**: Status bar not visible unless window is maximized
- **Solution**: Browser must be launched with `--start-maximized`
- **Warning**: Do NOT use zoom (`document.body.style.zoom`) - breaks maximized state

### Menu Already Open
- **Issue**: Clicking menu header when submenu is already open will close it
- **Solution**: Always check if submenu is visible before clicking menu header

### Timing Issues
- **Issue**: UI animations can cause elements to not be immediately clickable
- **Solution**: Use appropriate delays (0.5s for menu animations, 1s for sidebar)

### Selector Timeouts
- **Current**: 3000ms (3 seconds) default timeout
- **Note**: UI is fast, long timeouts usually indicate wrong selector

## Selector Priority Strategy

When finding elements, use this priority order:
1. `data-testid` attributes (most reliable)
2. `aria-label` attributes (good for accessibility)
3. Class selectors with specific names
4. Text content matching (`:has-text()`)
5. CSS path as last resort

## Testing Selectors

To test a selector in browser console:
```javascript
// Check if element exists
document.querySelector('.workflows-tab-button')

// Check if element is visible
const elem = document.querySelector('.workflows-tab-button');
elem && window.getComputedStyle(elem).display !== 'none'

// Find all matching elements
document.querySelectorAll('.p-menubar-submenu li')
```

## Updates and Maintenance

When UI changes occur:
1. Test selector in browser DevTools first
2. Update this document
3. Update `src/utils/selectors.py`
4. Test with actual MCP tools
5. Document any timing requirements