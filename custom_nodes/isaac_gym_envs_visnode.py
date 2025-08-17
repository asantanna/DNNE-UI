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
            dnne_config = None
            if 'env' in task_config and 'dnne' in task_config['env']:
                dnne_config = task_config['env']['dnne']
            elif 'dnne' in task_config:
                dnne_config = task_config['dnne']
            
            if dnne_config:
                schema_info['schema_levels'] = dnne_config.get('schema_levels', [])
                schema_info['nested_schemas'] = dnne_config.get('nested_schemas', {})
                
                # Load options and defaults for each level
                for level in schema_info['schema_levels']:
                    # Get options
                    options_key = f"{level}_options"
                    if options_key in dnne_config:
                        schema_info['level_options'][level] = dnne_config[options_key]
                    
                    # Get default value
                    default_key = f"default_{level}"
                    if default_key in dnne_config:
                        schema_info['defaults'][level] = dnne_config[default_key]
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
                options = schema_info['level_options'].get(level_name)
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
        schema_info = cls.get_task_schema_info(task)
        
        if not schema_info['nested_schemas']:
            return "No schema available for this task"
        
        # Navigate to the correct schema based on selections
        current_schema = schema_info['nested_schemas']
        path_parts = []
        
        # For hierarchical schemas, navigate the tree
        for level in schema_info['schema_levels']:
            if level in selections and selections[level]:
                level_value = selections[level]
                if isinstance(current_schema, dict) and level_value in current_schema:
                    current_schema = current_schema[level_value]
                    path_parts.append(level_value)
                else:
                    return f"Schema not found for: {' > '.join(path_parts + [level_value])}"
        
        # Format the schema display
        lines = []
        
        # Add path if hierarchical
        if path_parts:
            lines.append(f"Schema: {' > '.join(path_parts)}")
            lines.append("=" * 40)
        
        # Add observation schema
        if isinstance(current_schema, dict):
            obs_count = current_schema.get('numObservations', 0)
            obs_schema = current_schema.get('observationSchema', {})
            
            if obs_count:
                lines.append(f"\nObservations:")
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
            
            # Add action schema
            act_count = current_schema.get('numActions', 0)
            act_schema = current_schema.get('actionSchema', {})
            
            if act_count:
                lines.append(f"\nActions:")
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
            description = current_schema.get('description')
            if description:
                lines.append(f"\nDescription: {description}")
        
        result = "\n".join(lines) if lines else "No schema details available"
        return result
    
    # Standard node interface
    RETURN_TYPES = ("ISAAC_ENV_CONFIG_PYDICT",)
    RETURN_NAMES = ("env",)
    FUNCTION = None  # Virtual node - no execution
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs including dynamic selections"""
        task = kwargs.get("task")
        if not task:
            return "Task selection is required"
        
        # Get schema info for validation
        schema_info = cls.get_task_schema_info(task)
        
        # Validate dynamic widget selections
        for i, level in enumerate(schema_info['schema_levels']):
            widget_key = f"dynamic_{i+1}"
            selected = kwargs.get(widget_key)
            
            if selected and selected != "none":
                valid_options = schema_info['level_options'].get(level)
                if valid_options and selected not in valid_options:
                    return f"Invalid {level}: '{selected}'. Valid options: {valid_options}"
        
        # DNNE environment validation
        loader = IsaacGymConfigLoader()
        if loader.is_dnne_environment(task):
            num_envs = kwargs.get("num_envs", 1)
            if num_envs != 1:
                return f"DNNE environment '{task}' must use num_envs=1, got {num_envs}"
        
        return True
    
    @classmethod
    async def handle_widget_callback(cls, message):
        """Handle widget callbacks for this node type"""
        import json
        import logging
        
        widget_id = message.get("widget_id", "")
        widget_name = widget_id.split(".")[1] if "." in widget_id else ""
        event = message.get("event", "")
        event_params = message.get("event_params", {})
        
        logging.debug(f"[IsaacGymEnvsNode] Handling callback: {widget_name} - {event}")
        
        # Handle task widget callbacks
        if widget_name == "task":
            if event == "onChange":
                new_task = event_params.get("value")
                node_id = event_params.get("node_id")
                
                # Get schema info for the new task
                schema_info = cls.get_task_schema_info(new_task)
                
                # Format the schema display for the new task with defaults
                schema_display_text = cls.format_schema_display(new_task, schema_info.get('defaults', {}))
                
                # Generate JavaScript to update dynamic widgets
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
                node_id = event_params.get("node_id")
                initial_value = event_params.get("initial_value")
                if not initial_value:
                    # If no initial value, don't initialize
                    return {
                        "type": "widget_callback_response",
                        "widget_id": widget_id,
                        "chain": True
                    }
                
                # Get initial schema info
                schema_info = cls.get_task_schema_info(initial_value)
                
                # Format the schema display for initial task with defaults
                schema_display_text = cls.format_schema_display(initial_value, schema_info.get('defaults', {}))
                
                js_code = f"""
                // Initialize dynamic widgets on load
                const targetNode = app.graph.getNodeById({node_id});
                if (targetNode) {{
                    const schemaInfo = {json.dumps(schema_info)};
                    
                    // Initialize dynamic widgets
                    for (let i = 1; i <= {cls.MAX_DYNAMIC_LEVELS}; i++) {{
                        const widgetName = 'dynamic_' + i;
                        const widget = targetNode.widgets.find(w => w.name === widgetName);
                        
                        if (widget) {{
                            const levelIndex = i - 1;
                            if (levelIndex < schemaInfo.schema_levels.length) {{
                                const level = schemaInfo.schema_levels[levelIndex];
                                const options = schemaInfo.level_options[level];
                                const defaultValue = schemaInfo.defaults[level];
                                
                                widget.label = level;
                                widget.options.values = options;
                                widget.value = defaultValue;
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
                node_id = event_params.get("node_id")
                node_data = event_params.get("node_data", {})
                
                # Get task and build selections from node_data
                task = node_data.get("task")
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
                    widget_value = node_data.get(f"dynamic_{i+1}")
                    if widget_value and widget_value != 'none':
                        selections[level] = widget_value
                
                # Format the updated schema display
                schema_display_text = cls.format_schema_display(task, selections)
                
                # Generate JavaScript to update schema_display widget
                js_code = f"""
                // Update schema display when dynamic widget changes
                const targetNode = app.graph.getNodeById({node_id});
                if (targetNode) {{
                    // Update schema_display widget
                    const schemaWidget = targetNode.widgets.find(w => w.name === 'schema_display');
                    if (schemaWidget) {{
                        schemaWidget.value = {json.dumps(schema_display_text)};
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