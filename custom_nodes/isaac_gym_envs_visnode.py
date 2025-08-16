"""
Isaac Gym Environments with Hierarchical Schema Support
Provides GPU-accelerated physics simulation environments with dynamic schema selection.

This version uses pre-allocated dynamic widgets with static indices for stability.
"""

import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
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
                    "default": "FrankaDNNE" if "FrankaDNNE" in task_list else "Cartpole",
                    "tooltip": "Select an IsaacGymEnvs task - REQUIRED for export",
                    "on_change": "update_dynamic_widgets"  # Callback hint for UI
                }),
            },
            "optional": {}
        }
        
        # Add pre-allocated dynamic widgets (indices 1, 2, 3)
        # These will be shown/hidden and relabeled based on task selection
        for i in range(1, cls.MAX_DYNAMIC_LEVELS + 1):
            widgets["optional"][f"dynamic_{i}"] = (["none"], {
                "default": "none",
                "tooltip": f"Dynamic selection level {i}",
                "hidden": True,  # Initially hidden
                "dynamic": True,  # Mark as dynamic for UI
                "on_change": "update_schema_display"  # Callback hint
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
            "height": 600,  # Doubled height in pixels
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
                options = schema_info['level_options'].get(level_name, [])
                default = schema_info['defaults'].get(level_name, options[0] if options else "")
                
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
                lines.append(f"\nObservations ({obs_count} elements):")
                for name, indices in obs_schema.items():
                    if isinstance(indices, list) and len(indices) == 2:
                        size = indices[1] - indices[0]
                        desc = f"{size} element{'s' if size > 1 else ''}"
                        lines.append(f"  • {name:<20} [{indices[0]:2}:{indices[1]:2}]  {desc}")
            
            # Add action schema
            act_count = current_schema.get('numActions', 0)
            act_schema = current_schema.get('actionSchema', {})
            
            if act_count:
                lines.append(f"\nActions ({act_count} elements):")
                for name, indices in act_schema.items():
                    if isinstance(indices, list) and len(indices) == 2:
                        size = indices[1] - indices[0]
                        desc = f"{size} element{'s' if size > 1 else ''}"
                        lines.append(f"  • {name:<20} [{indices[0]:2}:{indices[1]:2}]  {desc}")
            
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
                valid_options = schema_info['level_options'].get(level, [])
                if selected not in valid_options:
                    return f"Invalid {level}: '{selected}'. Valid options: {valid_options}"
        
        # DNNE environment validation
        loader = IsaacGymConfigLoader()
        if loader.is_dnne_environment(task):
            num_envs = kwargs.get("num_envs", 1)
            if num_envs != 1:
                return f"DNNE environment '{task}' must use num_envs=1, got {num_envs}"
        
        return True
    
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