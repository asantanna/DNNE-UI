"""UI element selectors for DNNE UI"""

# Sidebar tabs
WORKFLOWS_TAB = ".workflows-tab-button"
NODE_LIBRARY_TAB = ".node-library-tab-button"
MODEL_LIBRARY_TAB = ".model-library-tab-button"  # To be removed
QUEUE_TAB = ".queue-tab-button"  # To be removed
SIDEBAR_BUTTON = ".side-bar-button"  # Generic sidebar button
SIDEBAR_CONTAINER = ".side-bar-container"
SIDEBAR_BUTTON_ACTIVE = ".side-bar-button.active"
SIDEBAR_CONTENT_CONTAINER = ".sidebar-content-container"

# Action bar
EXPORT_BUTTON = '[data-testid="export-button"]'
EXPORT_TARGET_DROPDOWN = ".export-target-dropdown"
RUN_AFTER_EXPORT = "#run-after-export"  # TBD - need to verify actual selector
STOP_BUTTON = 'button[aria-label="Stop"]'
SHOW_LOGS_BUTTON = 'button[aria-label="Show Logs"]'
TARGET_DROPDOWN = ".target-dropdown"

# Canvas controls
ZOOM_IN = 'button[aria-label="Zoom In"]'
ZOOM_OUT = 'button[aria-label="Zoom Out"]'
FIT_VIEW = 'button[aria-label="Fit View"]'
SELECT_MODE = 'button[aria-label="Select Mode"]'

# Canvas/Graph elements
CANVAS = 'canvas'
GRAPH_CANVAS = '.graph-canvas'
CANVAS_LINKS = '.link, .connection, [class*="link"]'
CANVAS_NODES = '.node, [class*="node"]:not([class*="node-"])'
LINK_LINE = '.link-line'
CONNECTION_LINE = '.connection-line'
NODE_ELEMENT = '.node'
LITEGRAPH_NODE = '.litegraph'

# Workflow management
WORKFLOW_TABS = ".workflow-tabs"
WORKFLOW_ACTIVE_TAB = ".workflow-tabs .active-tab"
WORKFLOWS_TAB_SELECTED = ".workflows-tab-button.side-bar-button-selected"
CLOSE_WORKFLOW = ".close-workflow-button"
WORKFLOWS_TREE = ".tree-explorer"
WORKFLOW_SEARCH = 'input[placeholder*="Search Workflows"]'
WORKFLOW_JSON_ITEMS = 'li[aria-label*=".json"]'

# Menu bar
MENU_BAR = ".top-menubar"
COMFYUI_MENU = ".comfyui-menu"  # Main menu bar
MENU_ITEMS = ".p-menubar-item-link"
MENU_LABELS = ".p-menubar-item-label"
MENU_SUBMENU = ".p-menubar-submenu"

# Status bar
STATUS_BAR = ".status-bar"  # Main status bar (if it exists)
AGENT_STATUS = ".agent-status-bar"
AGENT_STATUS_BOTTOM = ".agent-status-bar-bottom"  # The actual agent status bar in the UI

# Dialogs
DIALOG = ".p-dialog"
DIALOG_HEADER = ".p-dialog-header"
DIALOG_CONTENT = ".p-dialog-content"
DIALOG_FOOTER = ".p-dialog-footer"
DIALOG_CLOSE = ".p-dialog-close-button"
DIALOG_BUTTONS = ".p-dialog-content button"  # Buttons within dialog content

# Toast notifications
TOAST_MESSAGE = ".p-toast-message"
TOAST_ERROR = ".p-toast-message-error"
TOAST_SUCCESS = ".p-toast-message-success"

# Client/Logs
CLIENT_DROPDOWN = "#client-dropdown"  # Client dropdown in action bar
LOG_PANEL = ".log-panel"  # TBD
SHOW_ALL_LOGS = 'button[aria-label="Show Logs"]'
CLEAR_LOGS = ".clear-logs"  # TBD
LOG_CLIENT_DROPDOWN = ".log-client-dropdown"
LOG_FILTER_DROPDOWN = ".log-filter-dropdown"

# Dropdown item selectors for finding options in opened dropdowns
DROPDOWN_ITEM_SELECTORS = [
    '.p-select-option',                      # PrimeVue 4 select option (correct class!)
    '[role="listbox"] [role="option"]',      # ARIA compliant dropdowns (works!)
    '[role="option"]',                       # ARIA option elements (works!)
    '.p-select-overlay .p-select-option',    # PrimeVue select overlay with option
    '.p-select-overlay .p-select-item',      # PrimeVue select overlay (older)
    '.p-dropdown-panel .p-dropdown-item',    # Alternative dropdown panel
    '.p-select-items li',                    # List items in select
    '.p-select-item',                        # Generic select items
    '.p-dropdown-item',                      # Generic dropdown items
    'option'                                 # Standard HTML select options
]

# Dropdown selector mapping for different UI locations
DROPDOWN_SELECTORS = {
    "taskbar": {
        "client": CLIENT_DROPDOWN  # #client-dropdown
    },
    "log_window": {
        "client": LOG_CLIENT_DROPDOWN,  # .log-client-dropdown
        "filter": LOG_FILTER_DROPDOWN   # .log-filter-dropdown
    }
}

# Button selector mapping for different UI locations
BUTTON_SELECTORS = {
    "taskbar": {
        "export": EXPORT_BUTTON,          # [data-testid="export-button"]
        "stop": STOP_BUTTON,              # button[aria-label="Stop"]
        "show_logs": SHOW_LOGS_BUTTON     # button[aria-label="Show Logs"]
    },
    "canvas": {
        "zoom_in": ZOOM_IN,               # button[aria-label="Zoom In"]
        "zoom_out": ZOOM_OUT,             # button[aria-label="Zoom Out"]
        "fit_view": FIT_VIEW,             # button[aria-label="Fit View"]
        "select_mode": SELECT_MODE        # button[aria-label="Select Mode"]
    },
    "dialog": {
        "close": DIALOG_CLOSE,            # .p-dialog-close-button
        "confirm": f"{DIALOG_FOOTER} button:has-text('Yes'), {DIALOG_FOOTER} button:has-text('Confirm')",
        "cancel": f"{DIALOG_FOOTER} button:has-text('No'), {DIALOG_FOOTER} button:has-text('Cancel')"
    },
    "sidebar": {
        "close_workflow": CLOSE_WORKFLOW  # .close-workflow-button
    },
    "log_window": {
        "clear": CLEAR_LOGS               # .clear-logs (if it exists)
    }
}

# Tab selector mapping
TAB_SELECTORS = {
    "workflows": WORKFLOWS_TAB,
    "nodes": NODE_LIBRARY_TAB,
    "node_library": NODE_LIBRARY_TAB,
    "models": MODEL_LIBRARY_TAB,  # To be removed
    "queue": QUEUE_TAB  # To be removed
}

# Menu indices for top-level menus
MENU_INDICES = {
    "workflow": 1,
    "edit": 2,
    "dnne": 3,
    "help": 4
}

# Submenu item indices (positions include separators)
SUBMENU_ITEMS = {
    "workflow": {
        "new": 1, "new blank workflow": 1,
        "open": 3, "open workflow": 3,
        "browse templates": 4,
        "save": 6, "save workflow": 6,
        "save as": 7, "save workflow as": 7,
        "export": 8, "export workflow": 8,
        "export api": 9, "export (api)": 9
    },
    "edit": {
        "undo": 1,
        "redo": 2,
        # separator at 3
        "refresh": 4, "refresh node definitions": 4,
        # separator at 5
        "clear": 6, "clear workflow": 6,
        # separator at 7
        "clipspace": 8, "open clipspace": 8
    },
    "dnne": {},  # To be populated based on actual menu items
    "help": {}   # To be populated based on actual menu items
}

def get_workflow_selector(name: str) -> str:
    """Get selector for a specific workflow in the tree"""
    return f'li[aria-label="{name}"] > .p-tree-node-content'

def get_menu_item_selector(index: int) -> str:
    """Get selector for menu item by index"""
    return f".p-menubar-root-list > li:nth-child({index})"

def get_submenu_item_selector(index: int) -> str:
    """Get selector for submenu item by index"""
    # More specific selector to avoid multiple matches - target visible submenu
    return f".p-menubar-submenu:not([style*='display: none']) li:nth-child({index}) .p-menubar-item-link"