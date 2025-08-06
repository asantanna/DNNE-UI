"""
Centralized JavaScript snippets for browser.evaluate() calls
This module provides a single source of truth for all JavaScript code executed in the browser.
"""

import json
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from browser_controller import BrowserController

# Dictionary of JavaScript snippet templates
JS_SNIPPETS = {
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
    
    "close_menu": """
        () => {{ document.body.click(); }}
    """,
    
    # ===== Canvas Operations =====
    "get_zoom_level": """
        () => {{
            if (window.app && window.app.canvas && window.app.canvas.ds) {{
                return window.app.canvas.ds.scale;
            }}
            return null;
        }}
    """,
    
    "get_node_count": """
        () => {{
            // Try to access the graph directly
            if (window.app && window.app.canvas && window.app.canvas.graph) {{
                const nodes = window.app.canvas.graph.nodes;
                return Array.isArray(nodes) ? nodes.length : 0;
            }}
            
            // Fallback to DOM query
            const nodeElements = document.querySelectorAll('.node, .litegraph, [class*="node"]');
            return nodeElements.length;
        }}
    """,
    
    "are_links_visible": """
        () => {{
            const links = document.querySelectorAll('.link, .connection, [class*="link"]');
            if (links.length > 0) {{
                const firstLink = links[0];
                const style = window.getComputedStyle(firstLink);
                return style.display !== 'none' && style.visibility !== 'hidden';
            }}
            return false;
        }}
    """,
    
    "get_canvas_state": """
        () => {{
            const state = {{
                has_canvas: false,
                zoom: 1.0,
                node_count: 0,
                selected_nodes: 0,
                links_visible: true,
                sidebar_open: false,
                workflow_name: null
            }};
            
            // Check for canvas
            const canvas = document.querySelector('canvas') || document.querySelector('.graph-canvas');
            state.has_canvas = !!canvas;
            
            // Get zoom level
            if (window.app && window.app.canvas && window.app.canvas.ds) {{
                state.zoom = window.app.canvas.ds.scale;
            }}
            
            // Get node count
            if (window.app && window.app.canvas && window.app.canvas.graph) {{
                const nodes = window.app.canvas.graph.nodes;
                state.node_count = Array.isArray(nodes) ? nodes.length : 0;
                
                // Count selected nodes
                if (Array.isArray(nodes)) {{
                    state.selected_nodes = nodes.filter(n => n.selected).length;
                }}
            }} else {{
                const nodeElements = document.querySelectorAll('.node');
                state.node_count = nodeElements.length;
            }}
            
            // Check links visibility
            const links = document.querySelectorAll('.link, .connection');
            if (links.length > 0) {{
                const firstLink = links[0];
                const style = window.getComputedStyle(firstLink);
                state.links_visible = style.display !== 'none' && style.visibility !== 'hidden';
            }}
            
            // Check sidebar
            const sidebar = document.querySelector('.side-bar-container');
            state.sidebar_open = sidebar && !sidebar.classList.contains('collapsed');
            
            // Get workflow name
            const activeTab = document.querySelector('.workflow-tabs .active-tab');
            if (activeTab) {{
                state.workflow_name = activeTab.textContent?.trim();
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
    
    # ===== Export Operations =====
    "is_checkbox_checked": """
        () => {{
            const cb = document.querySelector('{selector}');
            return cb ? cb.checked : false;
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
    
    "get_export_status": """
        () => {{
            // Look for status messages in various places
            const statusBar = document.querySelector('.status-bar');
            const toastSuccess = document.querySelector('.p-toast-message-success');
            const toastError = document.querySelector('.p-toast-message-error');
            const dialogMessage = document.querySelector('.p-dialog-content');
            
            if (toastSuccess) {{
                return {{ success: true, message: toastSuccess.textContent?.trim() }};
            }}
            if (toastError) {{
                return {{ success: false, message: toastError.textContent?.trim() }};
            }}
            if (dialogMessage) {{
                return {{ success: null, message: dialogMessage.textContent?.trim() }};
            }}
            if (statusBar) {{
                return {{ success: null, message: statusBar.textContent?.trim() }};
            }}
            return {{ success: null, message: null }};
        }}
    """,
    
    "was_export_successful": """
        () => {{
            // Check for success toast
            const toast = document.querySelector('.p-toast-message-success');
            if (toast && toast.textContent?.includes('Export')) {{
                return true;
            }}
            
            // Check for error indicators
            const errorToast = document.querySelector('.p-toast-message-error');
            const errorDialog = document.querySelector('.p-dialog-title');
            if (errorToast || (errorDialog && errorDialog.textContent?.includes('Error'))) {{
                return false;
            }}
            
            return null; // Unknown state
        }}
    """,
    
    # ===== UI State Operations =====
    "get_sidebar_state": """
        () => {{
            const sidebar = document.querySelector('.side-bar-container');
            const isOpen = sidebar && !sidebar.classList.contains('collapsed');
            
            // Get active tab
            let activeTab = null;
            const tabs = document.querySelectorAll('.sidebar-tab');
            tabs.forEach(tab => {{
                if (tab.classList.contains('active')) {{
                    activeTab = tab.textContent?.trim();
                }}
            }});
            
            return {{
                isOpen: isOpen,
                activeTab: activeTab
            }};
        }}
    """,
    
    "get_agent_status": """
        () => {{
            const statusBar = document.querySelector('.agent-status-bar');
            if (!statusBar) {{
                return {{ connected: false, status: 'Unknown' }};
            }}
            
            const indicator = statusBar.querySelector('.status-indicator');
            const text = statusBar.textContent?.trim();
            const isConnected = indicator && indicator.classList.contains('connected');
            
            return {{
                connected: isConnected,
                status: text || 'Unknown'
            }};
        }}
    """,
    
    "get_ui_info": """
        () => {{
            const nodes = document.querySelectorAll('.node');
            const canvas = document.querySelector('canvas') || document.querySelector('.graph-canvas');
            const sidebar = document.querySelector('.side-bar-container');
            
            return {{
                nodeCount: nodes.length,
                hasCanvas: !!canvas,
                sidebarOpen: sidebar && !sidebar.classList.contains('collapsed'),
                url: window.location.href,
                title: document.title
            }};
        }}
    """,
    
    "get_page_state": """
        () => {{
            return {{
                url: window.location.href,
                workflow: document.title
            }};
        }}
    """,
    
    # ===== Error Handling =====
    "get_error_state": """
        () => {{
            return {{
                url: window.location.href,
                title: document.title,
                hasDialog: !!document.querySelector('.p-dialog'),
                hasToast: !!document.querySelector('.p-toast'),
                errorText: document.querySelector('.p-dialog-content')?.textContent || 
                          document.querySelector('.p-toast-message-error')?.textContent || null
            }};
        }}
    """,
    
    # ===== Utility Operations =====
    "test_connection": """
        () => {{ return 1 + 1; }}
    """,
    
    "get_document_title": """
        () => {{ return document.title; }}
    """,
    
    "find_elements_by_text": """
        () => {{
            const searchText = "{text}";
            const maxResults = {limit};
            const results = [];
            
            const walker = document.createTreeWalker(
                document.body,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );
            
            let node;
            while (node = walker.nextNode()) {{
                if (node.nodeValue && node.nodeValue.includes(searchText)) {{
                    const parent = node.parentElement;
                    if (parent) {{
                        results.push({{
                            tag: parent.tagName,
                            className: parent.className,
                            id: parent.id,
                            text: node.nodeValue.trim().substring(0, 100)
                        }});
                        
                        if (results.length >= maxResults) break;
                    }}
                }}
            }}
            
            return results;
        }}
    """,
    
    "execute_custom_code": """
        () => {{ {code} }}
    """
}


def get_javascript_snippet(name: str, params: Dict[str, Any] = None) -> str:
    """
    Get a JavaScript snippet by name and substitute parameters.
    
    Args:
        name: The name of the snippet to retrieve
        params: Dictionary of parameters to substitute into the snippet
        
    Returns:
        The formatted JavaScript code ready for execution
        
    Raises:
        KeyError: If the snippet name doesn't exist
        ValueError: If required parameters are missing
    """
    if name not in JS_SNIPPETS:
        raise KeyError(f"JavaScript snippet '{name}' not found. Available snippets: {list(JS_SNIPPETS.keys())}")
    
    snippet = JS_SNIPPETS[name]
    
    if params:
        try:
            # For JSON serializable params, convert to JSON string
            formatted_params = {}
            for key, value in params.items():
                if isinstance(value, (dict, list)):
                    formatted_params[key] = json.dumps(value)
                elif isinstance(value, bool):
                    formatted_params[key] = str(value).lower()
                else:
                    formatted_params[key] = value
            
            # Format the snippet with parameters
            snippet = snippet.format(**formatted_params)
        except KeyError as e:
            raise ValueError(f"Missing required parameter for snippet '{name}': {e}")
    
    return snippet


def list_snippets() -> Dict[str, str]:
    """
    Get a list of all available snippet names with brief descriptions.
    
    Returns:
        Dictionary mapping snippet names to their first line of documentation
    """
    descriptions = {}
    for name, code in JS_SNIPPETS.items():
        # Extract first meaningful line as description
        lines = [line.strip() for line in code.strip().split('\n') if line.strip() and not line.strip().startswith('()')]
        if lines and lines[0].startswith('//'):
            descriptions[name] = lines[0][2:].strip()
        else:
            descriptions[name] = name.replace('_', ' ').title()
    
    return descriptions


async def run_js_snippet_in_browser(browser, name: str, params: Dict[str, Any] = None) -> Any:
    """
    Execute a JavaScript snippet in the browser and return the result.
    
    This is a convenience function that combines get_javascript_snippet() and browser.evaluate()
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
    js_code = get_javascript_snippet(name, params)
    return await browser.evaluate(js_code)