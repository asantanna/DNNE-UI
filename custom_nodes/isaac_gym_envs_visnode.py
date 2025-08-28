"""
Isaac Gym Environments with Hierarchical Schema Support
Provides GPU-accelerated physics simulation environments with dynamic schema selection.

This version uses pre-allocated dynamic widgets with static indices for stability.
"""

import yaml
import os
import sys
from typing import Dict
from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors
from custom_nodes.utils.dnne_decorator import dnne_node
from .utils.isaac_gym_config_loader import IsaacGymEnvConfigLoader as IsaacGymConfigLoader

# Import for getting YAML path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dnne_config import get_isaac_gym_envs_path

@dnne_node(is_virtual=True)
class IsaacGymEnvsNode(RoboticsNodeBase):
    """Isaac Gym Environments with Hierarchical Schema Support
    Provides GPU-accelerated physics simulation environments with dynamic schema selection."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("utility")["color"] 
    BGCOLOR = get_node_colors("utility")["bgcolor"]
    CATEGORY = "robotics"
    
    # Maximum number of dynamic hierarchy levels we support
    MAX_DYNAMIC_LEVELS = 3
    
    def __init__(self):
        super().__init__()
        # Load available tasks
        loader = IsaacGymConfigLoader()
        self.available_tasks = loader.get_available_tasks()
    
    @classmethod
    def extract_null_action_from_schema(cls, schema_info: Dict, selections: Dict) -> tuple:
        """Extract nullAction from nested schema based on selections.
        
        Args:
            schema_info: The schema info dict from get_task_schema_info
            selections: Dict mapping level names to selected values
            
        Returns:
            tuple: (null_action_list, null_action_str) where:
                - null_action_list is the array of values or None
                - null_action_str is comma-separated string or empty string
        """
        # Navigate to the selected schema
        current_schema = schema_info.get('nested_schemas', {})
        for level in schema_info.get('schema_levels', []):
            if level in selections:
                value = selections[level]
                if isinstance(current_schema, dict) and value in current_schema:
                    current_schema = current_schema[value]
                else:
                    break
        
        # Extract nullAction
        if isinstance(current_schema, dict) and 'nullAction' in current_schema:
            null_action = current_schema['nullAction']
            # Format as comma-separated string for the widget
            null_action_str = ', '.join(str(x) for x in null_action)
            return null_action, null_action_str
        
        return None, ""
    
    @classmethod
    def get_task_schema_info(cls, task: str) -> Dict:
        """Load schema information for a specific task"""
        isaacgym_envs_path = get_isaac_gym_envs_path()
        task_cfg_path = isaacgym_envs_path / 'isaacgymenvs' / 'cfg' / 'task' / f'{task}.yaml'
        
        schema_info = {
            'schema_levels': [],
            'nested_schemas': {},
            'level_options': {},  # Store options for each level
            'defaults': {}
        }
        
        if not task_cfg_path.exists():
            return schema_info
            
        try:
            with open(task_cfg_path, 'r') as f:
                task_config = yaml.safe_load(f)
            
            # Look for dnne section - check both root level and env level
            dnne_section = None
            if 'env' in task_config and 'dnne' in task_config['env']:
                dnne_section = task_config['env']['dnne']
            
            if dnne_section:
                # Required fields - fail if missing
                if 'schema_levels' not in dnne_section:
                    raise ValueError(f"Task {task}: dnne section missing required 'schema_levels' field")
                if 'nested_schemas' not in dnne_section:
                    raise ValueError(f"Task {task}: dnne section missing required 'nested_schemas' field")
                
                schema_info['schema_levels'] = dnne_section['schema_levels']
                schema_info['nested_schemas'] = dnne_section['nested_schemas']
                
                # Load options and defaults for each level
                for level in schema_info['schema_levels']:
                    # Get options
                    options_key = f"{level}_options"
                    if options_key in dnne_section:
                        schema_info['level_options'][level] = dnne_section[options_key]
                    
                    # Get default value
                    default_key = f"default_{level}"
                    if default_key in dnne_section:
                        schema_info['defaults'][level] = dnne_section[default_key]
            else:
                # No DNNE config - check for basic env info
                if 'env' in task_config:
                    env_config = task_config['env']
                    basic_schema = {}
                    
                    if 'numObservations' in env_config:
                        basic_schema['numObservations'] = env_config['numObservations']
                    if 'numActions' in env_config:
                        basic_schema['numActions'] = env_config['numActions']
                    
                    if basic_schema:
                        schema_info['nested_schemas'] = basic_schema
                        
        except Exception as e:
            print(f"Warning: Could not load schema for {task}: {e}")
        
        return schema_info
    
    @classmethod
    def INPUT_TYPES(cls):
        # Create instance to get available tasks
        temp_instance = cls()
        task_list = temp_instance.available_tasks
        
        # Start with task widget (index 0)
        widgets = {
            "required": {
                "task": (task_list, {
                    # No default - will be set by onLoad callback
                    "tooltip": "Select an IsaacGymEnvs task - REQUIRED for export",
                    "widgetType": "DNNE_COMBO",  # Use new generic widget
                    "widget_id": "IsaacGymEnvsNode.task",
                    "listen_to": ["onChange", "onLoad"]
                }),
            },
            "optional": {}
        }
        
        # Add pre-allocated dynamic widgets (indices 1, 2, 3)
        # These will be shown/hidden and relabeled based on task selection
        for i in range(1, cls.MAX_DYNAMIC_LEVELS + 1):
            widgets["optional"][f"dynamic_{i}"] = (["none"], {
                "tooltip": f"Dynamic selection level {i}",
                "hidden": True,  # Initially hidden
                "dynamic": True,  # Mark as dynamic for UI
                "widgetType": "DNNE_COMBO",  # Use callback-enabled widget
                "widget_id": f"IsaacGymEnvsNode.dynamic_{i}",
                "listen_to": ["onChange"]  # Only onChange - task.onLoad handles initialization
            })
        
        # Add standard fixed widgets (starting from index 4)
        widgets["optional"].update({
            "dt": ("FLOAT", {
                "default": 0.01667,  # 60 Hz default
                "min": 0.001,
                "max": 0.1,
                "step": 0.001,
                "tooltip": "Simulation timestep (seconds)",
                "dnne_only": True  # Only for DNNE environments
            }),
            "num_envs": ("INT", {
                "default": 64,
                "min": 1,
                "max": 8192,
                "step": 1,
                "tooltip": "Number of parallel environments",
                "dnne_hide": True  # Hidden for DNNE environments
            }),
            "seed": ("INT", {
                "default": 42,
                "min": 0,
                "max": 1000000,
                "tooltip": "Random seed for reproducibility"
            }),
            "seed_control": (["fixed", "randomize", "increment", "decrement"], {
                "default": "fixed",
                "tooltip": "How to handle seed between runs"
            }),
            "headless": ("BOOLEAN", {
                "default": True,
                "tooltip": "Run in headless mode (no rendering)"
            }),
            "graphics_device_id": ("INT", {
                "default": 0,
                "min": 0,
                "max": 7,
                "tooltip": "GPU device ID for rendering"
            }),
            "sim_device": ("STRING", {
                "default": "cuda:0",
                "tooltip": "Device for physics simulation (e.g., cuda:0, cpu)"
            }),
            "physics_engine": (["physx", "flex"], {
                "default": "physx",
                "tooltip": "Physics engine backend"
            }),
            "multi_gpu": ("BOOLEAN", {
                "default": False,
                "tooltip": "Use multi-GPU simulation"
            }),
            "enable_cameras": ("BOOLEAN", {
                "default": False,
                "tooltip": "Enable camera sensors (impacts performance)"
            }),
            "force_render": ("BOOLEAN", {
                "default": False,
                "tooltip": "Force rendering even in headless mode"
            }),
            "use_gpu_pipeline": ("BOOLEAN", {
                "default": True,
                "tooltip": "Use GPU pipeline for faster training"
            }),
            "num_threads": ("INT", {
                "default": 0,
                "min": 0,
                "max": 64,
                "tooltip": "Number of CPU threads (0 = auto)"
            }),
            "solver_type": ("INT", {
                "default": 1,
                "min": 0,
                "max": 2,
                "tooltip": "PhysX solver type (0=PGS, 1=TGS)"
            }),
            "num_subscenes": ("INT", {
                "default": 0,
                "min": 0,
                "max": 32,
                "tooltip": "Number of PhysX subscenes (0 = auto)"
            }),
        })
        
        # Add schema display widget (always last)
        widgets["optional"]["schema_display"] = ("STRING", {
            "multiline": True,
            "default": "",
            "tooltip": "Current observation and action schema",
            "readonly": True,  # Make it read-only
            "widgetHeight": 200  # Set to 200px height
        })
        
        return widgets
    
    @classmethod
    def update_widgets_for_task(cls, task: str) -> Dict:
        """
        Get widget update information for a specific task.
        This would be called by the UI when task changes.
        Returns updates for dynamic widgets.
        """
        schema_info = cls.get_task_schema_info(task)
        updates = {}
        
        # Update each dynamic widget based on schema_levels
        for i in range(1, cls.MAX_DYNAMIC_LEVELS + 1):
            widget_key = f"dynamic_{i}"
            
            if i <= len(schema_info['schema_levels']):
                # This widget should be visible
                level_name = schema_info['schema_levels'][i-1]
                # Options are required for visible levels
                if level_name not in schema_info['level_options']:
                    raise ValueError(f"Task {task}: Missing options for level '{level_name}'")
                options = schema_info['level_options'][level_name]
                # Default is optional - None is valid
                default = schema_info['defaults'].get(level_name)
                
                updates[widget_key] = {
                    'hidden': False,
                    'label': level_name,  # Relabel the widget
                    'choices': options,
                    'default': default,
                    'tooltip': f"Select {level_name}"
                }
            else:
                # This widget should be hidden
                updates[widget_key] = {
                    'hidden': True,
                    'label': f"dynamic_{i}",
                    'choices': ["none"],
                    'default': "none"
                }
        
        # Also update schema display
        updates['schema_display'] = {
            'value': cls.format_schema_display(task, schema_info['defaults'])
        }
        
        return updates
    
    @classmethod
    def format_schema_display(cls, task: str, selections: Dict[str, str]) -> str:
        """Format the schema for display based on current selections"""
        # Import yaml_schema_utils for navigation
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from export_system.utils import yaml_schema_utils
        
        # Load the raw YAML config
        isaacgym_envs_path = get_isaac_gym_envs_path()
        task_cfg_path = isaacgym_envs_path / 'isaacgymenvs' / 'cfg' / 'task' / f'{task}.yaml'
        
        if not task_cfg_path.exists():
            return "No schema available for this task"
        
        try:
            with open(task_cfg_path, 'r') as f:
                task_config = yaml.safe_load(f)
        except Exception as e:
            return f"Error loading task config: {e}"
        
        # Build level_values from selections
        level_values = {}
        schema_levels = yaml_schema_utils.get_dnne_schema_levels(task_config)
        for level in schema_levels:
            if level in selections and selections[level] and selections[level] != 'none':
                level_values[level] = selections[level]
        
        # Use utility function to navigate to the correct schema
        current_schema = yaml_schema_utils.get_nested_schema_value(task_config, level_values)
        
        if not current_schema:
            # Try getting basic env info if no nested schema
            env_config = yaml_schema_utils.navigate_schema(task_config, ['env'])
            if env_config and ('numObservations' in env_config or 'numActions' in env_config):
                current_schema = env_config
            else:
                return "No schema available for current selections"
        
        # Format the schema display
        lines = []
        
        # Add path if hierarchical
        if level_values:
            path_parts = [level_values.get(level, '') for level in schema_levels if level in level_values]
            if path_parts:
                lines.append(f"Schema: {' > '.join(path_parts)}")
                lines.append("=" * 40)
        
        # Add observation schema if available
        if 'numObservations' in current_schema:
            obs_count = current_schema['numObservations']
            obs_schema = current_schema.get('observationSchema', {})
            
            if obs_count:
                lines.append(f"\nObservations: ({obs_count} total)")
                if obs_schema:
                    for name, indices in obs_schema.items():
                        if isinstance(indices, (int, float)):
                            # Single element: display as [x]
                            lines.append(f"  • {name:<20} [{int(indices):2}]")
                        elif isinstance(indices, list) and len(indices) == 2:
                            # Range: display as [start-end] if different, [x] if same
                            if indices[0] == indices[1]:
                                lines.append(f"  • {name:<20} [{indices[0]:2}]")
                            else:
                                lines.append(f"  • {name:<20} [{indices[0]:2}-{indices[1]:2}]")
        
        # Add action schema if available
        if 'numActions' in current_schema:
            act_count = current_schema['numActions']
            act_schema = current_schema.get('actionSchema', {})
            
            if act_count:
                lines.append(f"\nActions: ({act_count} total)")
                if act_schema:
                    for name, indices in act_schema.items():
                        if isinstance(indices, (int, float)):
                            # Single element: display as [x]
                            lines.append(f"  • {name:<20} [{int(indices):2}]")
                        elif isinstance(indices, list) and len(indices) == 2:
                            # Range: display as [start-end] if different, [x] if same
                            if indices[0] == indices[1]:
                                lines.append(f"  • {name:<20} [{indices[0]:2}]")
                            else:
                                lines.append(f"  • {name:<20} [{indices[0]:2}-{indices[1]:2}]")
        
        # Add description if available
        if 'description' in current_schema:
            description = current_schema['description']
            lines.append(f"\nDescription: {description}")
        
        # If we have no meaningful info, say so
        if not lines:
            if current_schema:
                # We have a schema but no display info
                return f"Schema found but no display details available\n(Keys: {', '.join(current_schema.keys())})"
            else:
                return "No schema details available"
        
        return "\n".join(lines)
    
    # Standard node interface
    OUTPUT_DICT = {
        0: {"type": "ISAAC_ENV_CONFIG_PYDICT", "name": "env", "virtual": True}  # Virtual config output
    }
    # RETURN_TYPES and RETURN_NAMES auto-generated by OutputDictMixin
    FUNCTION = None  # Virtual node - no execution
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs including dynamic selections"""
        # Task is required
        if "task" not in kwargs:
            return "Task parameter is required"
        task = kwargs["task"]
        if not task:
            return "Task selection is required"
        
        # Get schema info for validation
        schema_info = cls.get_task_schema_info(task)
        
        # Validate dynamic widget selections
        for i, level in enumerate(schema_info['schema_levels']):
            widget_key = f"dynamic_{i+1}"
            if widget_key not in kwargs:
                return f"Missing required parameter: {widget_key}"
            selected = kwargs[widget_key]
            
            if selected and selected != "none":
                # Validate options are available for this level
                if level not in schema_info['level_options']:
                    return f"No options available for level: {level}"
                valid_options = schema_info['level_options'][level]
                if selected not in valid_options:
                    return f"Invalid {level}: '{selected}'. Valid options: {valid_options}"
        
        # DNNE environment validation
        loader = IsaacGymConfigLoader()
        if loader.is_dnne_environment(task):
            # num_envs is required
            if "num_envs" not in kwargs:
                return "num_envs parameter is required"
            num_envs = kwargs["num_envs"]
            if num_envs != 1:
                return f"DNNE environment '{task}' must use num_envs=1, got {num_envs}"
        
        return True
    
    @classmethod
    async def handle_widget_callback(cls, message):
        """Handle widget callbacks for this node type"""
        import json
        import logging
        
        # Validate required message fields
        if "widget_id" not in message:
            raise ValueError("WebSocket message missing required 'widget_id' field")
        if "event" not in message:
            raise ValueError("WebSocket message missing required 'event' field")
        if "event_params" not in message:
            raise ValueError("WebSocket message missing required 'event_params' field")
        
        widget_id = message["widget_id"]
        widget_name = widget_id.split(".")[1] if "." in widget_id else ""
        event = message["event"]
        event_params = message["event_params"]
        
        logging.debug(f"[IsaacGymEnvsNode] Handling callback: {widget_name} - {event}")
        
        # Handle task widget callbacks
        if widget_name == "task":
            if event == "onChange":
                # Validate required event parameters
                if "value" not in event_params:
                    raise ValueError("onChange event missing required 'value' parameter")
                if "node_id" not in event_params:
                    raise ValueError("onChange event missing required 'node_id' parameter")
                
                new_task = event_params["value"]
                node_id = event_params["node_id"]
                
                # Get schema info for the new task
                schema_info = cls.get_task_schema_info(new_task)
                
                # Format the schema display for the new task with defaults
                defaults = schema_info.get('defaults', {})
                schema_display_text = cls.format_schema_display(new_task, defaults)
                
                # Extract nullAction using the utility method
                null_action, null_action_str = cls.extract_null_action_from_schema(schema_info, defaults)
                if null_action:
                    logging.debug(f"[IsaacGymEnvsNode] Task change - extracted nullAction: {null_action_str}")
                else:
                    logging.warning(f"[IsaacGymEnvsNode] No nullAction found for task '{new_task}' with defaults {defaults}")
                
                # Generate JavaScript to update dynamic widgets AND connected nodes
                js_code = f"""
                // Update dynamic widgets for task: {new_task}
                const targetNode = app.graph.getNodeById({node_id});
                if (targetNode) {{
                    const schemaInfo = {json.dumps(schema_info)};
                    
                    // Update dynamic widgets based on schema
                    for (let i = 1; i <= {cls.MAX_DYNAMIC_LEVELS}; i++) {{
                        const widgetName = 'dynamic_' + i;
                        const widget = targetNode.widgets.find(w => w.name === widgetName);
                        
                        if (widget) {{
                            const levelIndex = i - 1;
                            if (levelIndex < schemaInfo.schema_levels.length) {{
                                const level = schemaInfo.schema_levels[levelIndex];
                                const options = schemaInfo.level_options[level];
                                const defaultValue = schemaInfo.defaults[level];
                                
                                // Update widget
                                widget.label = level;
                                widget.options.values = options;
                                widget.value = defaultValue;
                                widget.hidden = false;
                            }} else {{
                                // Hide unused widgets
                                widget.hidden = true;
                                widget.value = 'none';
                            }}
                        }}
                    }}
                    
                    // Update schema_display widget
                    const schemaWidget = targetNode.widgets.find(w => w.name === 'schema_display');
                    if (schemaWidget) {{
                        schemaWidget.value = {json.dumps(schema_display_text)};
                    }}
                    
                    // Update null_action in all connected IsaacGymSim nodes
                    const nullActionStr = {json.dumps(null_action_str)};
                    if (nullActionStr) {{
                        // Find nodes connected to this node's env output (output index 0)
                        const outputLinks = targetNode.outputs[0]?.links || [];
                        outputLinks.forEach(linkId => {{
                            const link = app.graph.links[linkId];
                            if (link) {{
                                const connectedNode = app.graph.getNodeById(link.target_id);
                                if (connectedNode && connectedNode.type === 'IsaacGymSim') {{
                                    // Update the null_action widget
                                    const nullActionWidget = connectedNode.widgets?.find(w => w.name === 'null_action');
                                    if (nullActionWidget) {{
                                        nullActionWidget.value = nullActionStr;
                                        // console.log(`[IsaacGymEnvs] Task change - Updated null_action in IsaacGymSim node ${{connectedNode.id}} to: "${{nullActionStr}}"`);
                                    }}
                                }}
                                // TODO: Also handle PPOAgent nodes that might be connected
                            }}
                        }});
                    }}
                    
                    // Update node size
                    targetNode.setSize(targetNode.computeSize());
                    
                    // Force immediate redraw with both dirty flags
                    app.graph.setDirtyCanvas(true, true);
                    
                    // Use requestAnimationFrame to ensure draw happens
                    requestAnimationFrame(() => {{
                        app.canvas.draw(true, true);
                    }});
                }}
                """
                
                return {
                    "type": "widget_callback_response",
                    "widget_id": widget_id,
                    "code_payload": js_code,
                    "chain": True
                }
            
            elif event == "onLoad":
                # Initialize widget on load
                if "node_id" not in event_params:
                    raise ValueError("onLoad event missing required 'node_id' parameter")
                if "initial_value" not in event_params:
                    raise ValueError("onLoad event missing required 'initial_value' parameter")
                if "node_data" not in event_params:
                    raise ValueError("onLoad event missing required 'node_data' parameter")
                    
                node_id = event_params["node_id"]
                initial_value = event_params["initial_value"]
                node_data = event_params["node_data"]
                
                if not initial_value:
                    # If no initial value, don't initialize
                    return {
                        "type": "widget_callback_response",
                        "widget_id": widget_id,
                        "chain": True
                    }
                
                # Get initial schema info
                schema_info = cls.get_task_schema_info(initial_value)
                
                # Extract loaded widget values from node_data
                loaded_widget_values = {}
                for i, level in enumerate(schema_info['schema_levels']):
                    widget_key = f"dynamic_{i+1}"
                    if widget_key in node_data:
                        widget_value = node_data[widget_key]
                        if widget_value and widget_value != 'none':
                            loaded_widget_values[level] = widget_value
                            logging.debug(f"[IsaacGymEnvsNode] onLoad: Found loaded value for {level}: {widget_value}")
                
                # Format schema display with loaded values
                schema_display_text = cls.format_schema_display(initial_value, loaded_widget_values)
                
                js_code = f"""
                // Initialize dynamic widgets on load
                const targetNode = app.graph.getNodeById({node_id});
                if (targetNode) {{
                    const schemaInfo = {json.dumps(schema_info)};
                    const loadedWidgetValues = {json.dumps(loaded_widget_values)};
                    
                    // Initialize dynamic widgets
                    for (let i = 1; i <= {cls.MAX_DYNAMIC_LEVELS}; i++) {{
                        const widgetName = 'dynamic_' + i;
                        const widget = targetNode.widgets.find(w => w.name === widgetName);
                        
                        if (widget) {{
                            const levelIndex = i - 1;
                            if (levelIndex < schemaInfo.schema_levels.length) {{
                                const level = schemaInfo.schema_levels[levelIndex];
                                const options = schemaInfo.level_options[level];
                                
                                // Update label and options
                                widget.label = level;
                                widget.options.values = options;
                                
                                // ALWAYS use loaded value - NO DEFAULTS!
                                const loadedValue = loadedWidgetValues[level];
                                widget.value = loadedValue;
                                // console.log(`[IsaacGymEnvs] onLoad: Set ${{level}} to loaded value: ${{loadedValue}}`);
                                
                                widget.hidden = false;
                            }} else {{
                                widget.hidden = true;
                            }}
                        }}
                    }}
                    
                    // Initialize schema_display widget
                    const schemaWidget = targetNode.widgets.find(w => w.name === 'schema_display');
                    if (schemaWidget) {{
                        schemaWidget.value = {json.dumps(schema_display_text)};
                    }}
                    
                    targetNode.setSize(targetNode.computeSize());
                    // Force immediate redraw with both dirty flags
                    app.graph.setDirtyCanvas(true, true);
                }}
                """
                
                return {
                    "type": "widget_callback_response",
                    "widget_id": widget_id,
                    "code_payload": js_code,
                    "chain": False
                }
        
        # Handle dynamic widget callbacks (dynamic_1, dynamic_2, dynamic_3)
        elif widget_name.startswith("dynamic_"):
            if event == "onChange":
                # Validate required event parameters
                if "node_id" not in event_params:
                    raise ValueError("onChange event missing required 'node_id' parameter")
                if "node_data" not in event_params:
                    raise ValueError("onChange event missing required 'node_data' parameter")
                    
                node_id = event_params["node_id"]
                node_data = event_params["node_data"]
                
                # Get task and build selections from node_data
                if "task" not in node_data:
                    raise ValueError("node_data missing required 'task' field")
                task = node_data["task"]
                if not task:
                    logging.error(f"Dynamic widget change without task in node_data")
                    return {
                        "type": "widget_callback_response",
                        "widget_id": widget_id,
                        "chain": True
                    }
                
                # Get schema info for the task
                schema_info = cls.get_task_schema_info(task)
                
                # Build selections from dynamic widget values
                selections = {}
                for i, level in enumerate(schema_info['schema_levels']):
                    widget_key = f"dynamic_{i+1}"
                    # Dynamic widgets should be present in node_data
                    if widget_key not in node_data:
                        raise ValueError(f"node_data missing required field: {widget_key}")
                    widget_value = node_data[widget_key]
                    if widget_value and widget_value != 'none':
                        selections[level] = widget_value
                
                # Format the updated schema display
                schema_display_text = cls.format_schema_display(task, selections)
                
                # Extract nullAction using the utility method
                null_action, null_action_str = cls.extract_null_action_from_schema(schema_info, selections)
                if null_action:
                    logging.debug(f"[IsaacGymEnvsNode] Dynamic widget change - extracted nullAction: {null_action_str}")
                else:
                    logging.warning(f"[IsaacGymEnvsNode] No nullAction found for task '{task}' with selections {selections}")
                
                # Generate JavaScript to update schema_display widget AND connected nodes
                js_code = f"""
                // Update schema display when dynamic widget changes
                const targetNode = app.graph.getNodeById({node_id});
                if (targetNode) {{
                    // Update schema_display widget
                    const schemaWidget = targetNode.widgets.find(w => w.name === 'schema_display');
                    if (schemaWidget) {{
                        schemaWidget.value = {json.dumps(schema_display_text)};
                    }}
                    
                    // Update null_action in all connected IsaacGymSim nodes
                    const nullActionStr = {json.dumps(null_action_str)};
                    if (nullActionStr) {{
                        // Find nodes connected to this node's env output (output index 0)
                        const outputLinks = targetNode.outputs[0]?.links || [];
                        outputLinks.forEach(linkId => {{
                            const link = app.graph.links[linkId];
                            if (link) {{
                                const connectedNode = app.graph.getNodeById(link.target_id);
                                if (connectedNode && connectedNode.type === 'IsaacGymSim') {{
                                    // Update the null_action widget
                                    const nullActionWidget = connectedNode.widgets?.find(w => w.name === 'null_action');
                                    if (nullActionWidget) {{
                                        nullActionWidget.value = nullActionStr;
                                        // console.log(`[IsaacGymEnvs] Updated null_action in IsaacGymSim node ${{connectedNode.id}} to: "${{nullActionStr}}"`);
                                    }}
                                }}
                                // TODO: Also handle PPOAgent nodes that might be connected
                            }}
                        }});
                    }}
                    
                    // Force immediate redraw with both dirty flags
                    app.graph.setDirtyCanvas(true, true);
                    
                    // Use requestAnimationFrame to ensure draw happens
                    requestAnimationFrame(() => {{
                        app.canvas.draw(true, true);
                    }});
                }}
                """
                
                return {
                    "type": "widget_callback_response",
                    "widget_id": widget_id,
                    "code_payload": js_code,
                    "chain": True
                }
        
        # Default response
        return {
            "type": "widget_callback_response",
            "widget_id": widget_id,
            "chain": True
        }
    
    @classmethod
    def IS_DNNE_ENVIRONMENT(cls, task_name):
        """Check if a task is a DNNE environment."""
        loader = IsaacGymConfigLoader()
        return loader.is_dnne_environment(task_name)

# Node registration
NODE_CLASS_MAPPINGS = {
    "IsaacGymEnvs": IsaacGymEnvsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IsaacGymEnvs": "Isaac Gym Environment Config"
}