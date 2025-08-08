"""
Centralized JavaScript snippets for browser automation.

ARCHITECTURAL RULES:
1. There should be NO JavaScript evaluate() calls outside this file.
   All browser.evaluate() calls must go through the js_* functions defined here.
2. Functions in this file must use the snippet mechanism (_JS_SNIPPETS dictionary),
   not inline JavaScript strings.

This module provides the single source of truth for all JavaScript code executed in the browser.
"""

import json
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from browser_controller import BrowserController


# ===== Public API Functions =====
# All functions below provide the public interface for JavaScript execution.
# They all follow the js_* naming convention.

async def js_is_dropdown_open(browser, selectors) -> bool:
    """Check if a dropdown is currently open."""
    return await _run_js_snippet_in_browser(browser, "is_dropdown_open", {"selectors": json.dumps(selectors)})

async def js_get_dropdown_items(browser, selectors) -> Dict[str, Any]:
    """Get dropdown items and their state."""
    return await _run_js_snippet_in_browser(browser, "get_dropdown_items", {"selectors": json.dumps(selectors)})

async def js_get_dropdown_item_details(browser, selectors) -> list:
    """Get detailed information about dropdown items."""
    return await _run_js_snippet_in_browser(browser, "get_dropdown_item_details", {"selectors": json.dumps(selectors)})

async def js_click_dropdown_item_by_index(browser, selector: str, index: int) -> bool:
    """Click a dropdown item by its index."""
    return await _run_js_snippet_in_browser(browser, "click_dropdown_item_by_index", {"selector": selector, "index": index})

async def js_click_menu_item_by_text(browser, item_text: str) -> bool:
    """Click a menu item by its text content."""
    return await _run_js_snippet_in_browser(browser, "click_menu_item_by_text", {"item_text": item_text})

async def js_get_canvas_state(browser) -> Dict[str, Any]:
    """Get comprehensive canvas state information."""
    return await _run_js_snippet_in_browser(browser, "get_canvas_state")

async def js_get_workflow_list(browser) -> list:
    """Get list of workflows from the sidebar."""
    return await _run_js_snippet_in_browser(browser, "get_workflow_list")

async def js_get_current_workflow_name(browser) -> str:
    """Get the name of the currently active workflow."""
    return await _run_js_snippet_in_browser(browser, "get_current_workflow_name")

async def js_check_workflow_element(browser, selector: str) -> Dict[str, Any]:
    """Check the state of a workflow element."""
    return await _run_js_snippet_in_browser(browser, "check_workflow_element", {"selector": selector})

async def js_click_confirm_button(browser) -> bool:
    """Click the confirm button in a dialog."""
    return await _run_js_snippet_in_browser(browser, "click_confirm_button")

async def js_trigger_save_shortcut(browser) -> None:
    """Trigger the save keyboard shortcut."""
    return await _run_js_snippet_in_browser(browser, "trigger_save_shortcut")

async def js_get_button_state(browser, selector: str) -> Dict[str, Any]:
    """Get the state of a button (exists, visible, disabled, text)."""
    return await _run_js_snippet_in_browser(browser, "get_button_state", {"selector": selector})

async def js_is_checkbox_checked(browser, selector: str) -> bool:
    """Check if a checkbox is checked."""
    return await _run_js_snippet_in_browser(browser, "is_checkbox_checked", {"selector": selector})

async def js_is_checkbox_disabled(browser, selector: str) -> bool:
    """Check if a checkbox is disabled."""
    return await _run_js_snippet_in_browser(browser, "is_checkbox_disabled", {"selector": selector})

async def js_set_checkbox(browser, selector: str, label: str, checked: bool) -> bool:
    """Set a checkbox state."""
    return await _run_js_snippet_in_browser(browser, "set_checkbox", {"selector": selector, "label": label, "checked": str(checked).lower()})

async def js_execute_custom_code(browser, code: str) -> Any:
    """Execute custom JavaScript code."""
    return await _run_js_snippet_in_browser(browser, "execute_custom_code", {"code": code})


# ===== Private Implementation =====

# Private dictionary of JavaScript snippet templates
_JS_SNIPPETS = {
    # ===== Dropdown Operations =====
    "is_dropdown_open": """
        () => {{
            const selectors = {selectors};
            for (const selector of selectors) {{
                const items = document.querySelectorAll(selector);
                if (items.length > 0 && items[0].offsetParent !== null) {{
                    return true;
                }}
            }}
            return false;
        }}
    """,
    
    "get_dropdown_items": """
        () => {{
            const selectors = {selectors};
            let items = [];
            
            // Try each selector until we find items
            for (const selector of selectors) {{
                const foundItems = document.querySelectorAll(selector);
                if (foundItems.length > 0) {{
                    items = foundItems;
                    break;
                }}
            }}
            
            const visible = items.length > 0 && items[0].offsetParent !== null;
            
            // Get item texts if visible - DO NOT MODIFY!
            const itemTexts = visible ? 
                Array.from(items).map(el => el.textContent?.trim()).filter(t => t) : [];
            
            return {{
                is_open: visible,
                item_count: itemTexts.length,
                items: itemTexts  // Keep original text with emojis
            }};
        }}
    """,
    
    "get_dropdown_item_details": """
        (() => {{
            const selectors = {selectors};
            const results = [];
            
            for (const selector of selectors) {{
                const options = document.querySelectorAll(selector);
                options.forEach((opt, index) => {{
                    const text = opt.textContent?.trim();
                    if (text) {{
                        results.push({{
                            text: text,
                            selector: selector,
                            index: index
                        }});
                    }}
                }});
                if (results.length > 0) break; // Use first selector that finds items
            }}
            
            return results;
        }})()
    """,
    
    "click_dropdown_item_by_index": """
        (() => {{
            const selector = "{selector}";
            const index = {index};
            const options = document.querySelectorAll(selector);
            if (options[index]) {{
                options[index].click();
                return true;
            }}
            return false;
        }})()
    """,
    
    # ===== Menu Operations =====
    "click_menu_item_by_text": """
        () => {{
            const items = document.querySelectorAll('.p-menubar-submenu .p-menubar-item-label');
            for (let item of items) {{
                if (item.textContent?.trim() === '{item_text}') {{
                    item.click();
                    return true;
                }}
            }}
            return false;
        }}
    """,
    
    # ===== Canvas Operations =====
    "get_canvas_state": """
        () => {{
            const canvas = document.querySelector('canvas');
            const nodeElements = document.querySelectorAll('.node, [class*="node"]:not([class*="node-"])');
            const linkElements = document.querySelectorAll('.link, .connection, [class*="link"]');
            
            // Try to get zoom from LiteGraph or app
            let zoom = 1;
            try {{
                if (window.app && window.app.canvas) {{
                    zoom = window.app.canvas.ds?.scale || 1;
                }} else if (window.LiteGraph && window.LiteGraph.canvas) {{
                    zoom = window.LiteGraph.canvas.ds?.scale || 1;
                }}
            }} catch(e) {{}}
            
            // Count nodes from LiteGraph if available
            let nodeCount = nodeElements.length;
            try {{
                if (window.app && window.app.graph) {{
                    nodeCount = window.app.graph._nodes?.length || nodeCount;
                }}
            }} catch(e) {{}}
            
            const state = {{
                has_canvas: !!canvas,
                canvas_visible: canvas ? (canvas.offsetParent !== null) : false,
                node_count: nodeCount,
                link_count: linkElements.length,
                zoom_level: zoom,
                // Try to get workflow info
                workflow: {{
                    has_nodes: nodeCount > 0,
                    is_empty: nodeCount === 0
                }}
            }};
            
            // Try to get selected nodes
            try {{
                if (window.app && window.app.canvas) {{
                    const selectedNodes = window.app.canvas.selected_nodes || [];
                    state.selected_nodes = Object.keys(selectedNodes).length;
                }}
            }} catch(e) {{
                state.selected_nodes = 0;
            }}
            
            return state;
        }}
    """,
    
    # ===== Workflow Operations =====
    "get_workflow_list": """
        () => {{
            const items = document.querySelectorAll('li[aria-label*=".json"]');
            return Array.from(items).map(item => 
                item.getAttribute('aria-label')?.replace('.json', '') || ''
            );
        }}
    """,
    
    "get_current_workflow_name": """
        () => {{
            // Look for active workflow tab
            const activeTab = document.querySelector('.workflow-tabs .active-tab');
            if (activeTab) {{
                return activeTab.textContent?.trim();
            }}
            
            // Fallback to document title
            const title = document.title;
            if (title && title !== 'DNNE UI' && !title.includes('Unsaved')) {{
                return title;
            }}
            
            // Check for workflow name in UI
            const workflowNameElem = document.querySelector('.workflow-name');
            if (workflowNameElem) {{
                return workflowNameElem.textContent?.trim();
            }}
            
            return null;
        }}
    """,
    
    "check_workflow_element": """
        () => {{
            const elem = document.querySelector('{selector}');
            return {{
                exists: !!elem,
                visible: elem ? elem.offsetParent !== null : false,
                text: elem ? elem.textContent?.trim() : null
            }};
        }}
    """,
    
    "click_confirm_button": """
        () => {{
            const buttons = document.querySelectorAll('.p-dialog-content button');
            for (const button of buttons) {{
                if (button.textContent.trim() === 'Confirm') {{
                    button.click();
                    return true;
                }}
            }}
            return false;
        }}
    """,
    
    "trigger_save_shortcut": """
        () => {{
            document.dispatchEvent(new KeyboardEvent('keydown', {{
                key: 's',
                ctrlKey: true,
                bubbles: true
            }}));
        }}
    """,
    
    # ===== Button Operations =====
    "get_button_state": """
        () => {{
            const button = document.querySelector('{selector}');
            if (!button) return {{ exists: false, visible: false, disabled: false, text: null }};
            
            return {{
                exists: true,
                visible: button.offsetParent !== null,
                disabled: button.disabled || button.hasAttribute('disabled'),
                text: button.textContent?.trim() || button.getAttribute('aria-label') || null
            }};
        }}
    """,
    
    # ===== Export Operations =====
    "is_checkbox_checked": """
        () => {{
            const cb = document.querySelector('{selector}');
            return cb ? cb.checked : false;
        }}
    """,
    
    "is_checkbox_disabled": """
        () => {{
            const cb = document.querySelector('{selector}');
            return cb ? cb.disabled : false;
        }}
    """,
    
    "set_checkbox": """
        () => {{
            const checkboxes = document.querySelectorAll('{selector}');
            for (let cb of checkboxes) {{
                if (cb.parentElement && cb.parentElement.textContent?.includes('{label}')) {{
                    cb.checked = {checked};
                    cb.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    return true;
                }}
            }}
            return false;
        }}
    """,
    
    # ===== Custom Code Execution =====
    "execute_custom_code": """
        (() => {{
            {code}
        }})()
    """
}


def _get_javascript_snippet(name: str, params: Dict[str, Any] = None) -> str:
    """
    Get a JavaScript snippet template and substitute parameters.
    
    Args:
        name: The name of the snippet to retrieve
        params: Dictionary of parameters to substitute into the snippet
        
    Returns:
        The JavaScript code with parameters substituted
        
    Raises:
        KeyError: If the snippet name doesn't exist
        ValueError: If required parameters are missing
    """
    if name not in _JS_SNIPPETS:
        raise KeyError(f"JavaScript snippet '{name}' not found. Available snippets: {list(_JS_SNIPPETS.keys())}")
    
    snippet = _JS_SNIPPETS[name]
    
    if params:
        try:
            # Use format with double braces for JavaScript object literals
            # Single braces are used for Python string formatting
            return snippet.format(**params)
        except KeyError as e:
            raise ValueError(f"Missing required parameter for snippet '{name}': {e}")
    
    return snippet


def get_snippet_descriptions() -> Dict[str, str]:
    """
    Get a list of all available snippet names with brief descriptions.
    
    Returns:
        Dictionary mapping snippet names to their first line of documentation
    """
    descriptions = {}
    for name, code in _JS_SNIPPETS.items():
        # Extract first meaningful line as description
        lines = [line.strip() for line in code.strip().split('\n') if line.strip() and not line.strip().startswith('()')]
        if lines and lines[0].startswith('//'):
            descriptions[name] = lines[0][2:].strip()
        else:
            descriptions[name] = name.replace('_', ' ').title()
    
    return descriptions


async def _run_js_snippet_in_browser(browser, name: str, params: Dict[str, Any] = None) -> Any:
    """
    Execute a JavaScript snippet in the browser and return the result.
    
    This is a convenience function that combines _get_javascript_snippet() and browser.evaluate()
    into a single call for cleaner code.
    
    Args:
        browser: The BrowserController instance to execute the snippet in
        name: The name of the snippet to execute
        params: Dictionary of parameters to substitute into the snippet
        
    Returns:
        The result of executing the JavaScript snippet
        
    Raises:
        KeyError: If the snippet name doesn't exist
        ValueError: If required parameters are missing
    """
    js_code = _get_javascript_snippet(name, params)
    return await browser.evaluate(js_code)