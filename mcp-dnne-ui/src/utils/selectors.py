"""UI element selectors for DNNE UI"""

# Sidebar tabs
WORKFLOWS_TAB = ".workflows-tab-button"
NODE_LIBRARY_TAB = ".node-library-tab-button"
MODEL_LIBRARY_TAB = ".model-library-tab-button"  # To be removed
QUEUE_TAB = ".queue-tab-button"  # To be removed

# Action bar
EXPORT_BUTTON = '[data-testid="export-button"]'
EXPORT_TARGET_DROPDOWN = ".export-target-dropdown"
RUN_AFTER_EXPORT = "#run-after-export"  # TBD - need to verify actual selector

# Canvas controls
ZOOM_IN = '[aria-label="Zoom In"]'
ZOOM_OUT = '[aria-label="Zoom Out"]'
FIT_VIEW = '[aria-label="Fit View"]'
TOGGLE_LINKS = '[data-testid="toggle-link-visibility-button"]'

# Workflow management
WORKFLOW_TABS = ".workflow-tabs"
CLOSE_WORKFLOW = ".close-workflow-button"
WORKFLOWS_TREE = ".tree-explorer"
WORKFLOW_SEARCH = 'input[placeholder*="Search Workflows"]'

# Menu bar
MENU_BAR = ".top-menubar"
MENU_ITEMS = ".p-menubar-item-link"
MENU_LABELS = ".p-menubar-item-label"
MENU_SUBMENU = ".p-menubar-submenu"

# Status bar
STATUS_BAR = ".status-bar"  # TBD - need to verify actual selector
AGENT_STATUS = ".agent-status"  # TBD

# Dialogs
DIALOG = ".p-dialog"
DIALOG_HEADER = ".p-dialog-header"
DIALOG_CONTENT = ".p-dialog-content"
DIALOG_FOOTER = ".p-dialog-footer"
DIALOG_CLOSE = ".p-dialog-header-close"

# Client/Logs
CLIENT_DROPDOWN = ".client-dropdown"  # TBD
LOG_PANEL = ".log-panel"  # TBD
SHOW_ALL_LOGS = ".show-all-logs"  # TBD
CLEAR_LOGS = ".clear-logs"  # TBD

def get_workflow_selector(name: str) -> str:
    """Get selector for a specific workflow in the tree"""
    return f'li[aria-label="{name}"] .p-tree-node-content'

def get_menu_item_selector(index: int) -> str:
    """Get selector for menu item by index"""
    return f".p-menubar-root-list > li:nth-child({index})"

def get_submenu_item_selector(index: int) -> str:
    """Get selector for submenu item by index"""
    return f".p-menubar-submenu li:nth-child({index}) .p-menubar-item-link"