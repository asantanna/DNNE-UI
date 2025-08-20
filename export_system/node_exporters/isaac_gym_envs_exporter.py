#!/usr/bin/env python3
"""
Exporter for IsaacGymEnvs node with hierarchical schema support
Handles dynamic widget indices properly with pre-allocated slots
"""

import yaml
import os
import sys
from pathlib import Path
from ..graph_exporter import ExportableNode
from ..utils import yaml_schema_utils

# Add parent directory to path to import dnne_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dnne_config import get_isaac_gym_envs_path

class IsaacGymEnvsExporter(ExportableNode):
    """Exporter for IsaacGymEnvs virtual node with hierarchical schema support"""
    
    # Maximum number of dynamic levels (must match visnode)
    MAX_DYNAMIC_LEVELS = 3
    
    @classmethod
    def get_template_name(cls):
        # Virtual nodes don't need templates
        return None
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Virtual nodes don't generate code
        return {}
    
    @classmethod
    def get_imports(cls):
        return []
    
    @classmethod
    def get_output_names(cls):
        return ["env"]
    
    @classmethod
    def get_input_names(cls):
        return []
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        """Provide environment configuration schema including observation/action sizes"""
        # Widget indices are now static!
        # Index 0: task
        # Index 1-3: dynamic_1, dynamic_2, dynamic_3 (may be hidden)
        # Index 4+: fixed widgets (dt, num_envs, seed, etc.)
        
        # Get task first
        task_name = cls.get_node_parameter(node_data, 'task', widget_index=0)
        if not task_name:
            raise ValueError(
                f"IsaacGymEnvs node: 'task' parameter is required but was not found. "
                f"Available widgets: {node_data.get('widgets_values', [])[:5]}"
            )
        
        # Load schema information for this task
        schema_info = cls._load_task_schema(task_name)
        
        # Extract dynamic level values based on schema_levels
        level_values = {}
        for i, level in enumerate(schema_info.get('schema_levels', [])):
            if i < cls.MAX_DYNAMIC_LEVELS:
                # Dynamic widgets are at indices 1, 2, 3
                widget_value = cls.get_node_parameter(
                    node_data, 
                    f'dynamic_{i+1}', 
                    widget_index=i + 1
                )
                if widget_value and widget_value != 'none':
                    level_values[level] = widget_value
        
        # Navigate to the correct schema based on level selections
        current_schema = schema_info.get('nested_schemas', {})
        schema_path = []
        
        for level in schema_info.get('schema_levels', []):
            if level in level_values:
                value = level_values[level]
                if isinstance(current_schema, dict) and value in current_schema:
                    current_schema = current_schema[value]
                    schema_path.append(value)
                else:
                    # Schema not found, use defaults if available
                    break
        
        # Extract schema data
        observation_size = None
        action_size = None
        observation_schema = None
        action_schema = None
        
        if isinstance(current_schema, dict):
            observation_size = current_schema.get('numObservations')
            action_size = current_schema.get('numActions')
            observation_schema = current_schema.get('observationSchema')
            action_schema = current_schema.get('actionSchema')
        
        # Build output schema
        schema = {
            "outputs": {
                "env": {
                    "type": "env_config",
                    "task": task_name,
                    "schema_path": schema_path  # Include path for debugging
                }
            }
        }
        
        # Add level values
        for level, value in level_values.items():
            schema["outputs"]["env"][level] = value
        
        # Add size information if available
        if observation_size is not None:
            schema["outputs"]["env"]["observation_size"] = observation_size
            schema["outputs"]["env"]["flattened_size"] = observation_size  # For compatibility
        
        if action_size is not None:
            schema["outputs"]["env"]["action_size"] = action_size
        
        # Add semantic schemas if available
        if observation_schema is not None:
            schema["outputs"]["env"]["observation_schema"] = observation_schema
        
        if action_schema is not None:
            schema["outputs"]["env"]["action_schema"] = action_schema
        
        return schema
    
    @classmethod
    def _load_task_schema(cls, task_name):
        """Load schema information from task YAML using utility functions"""
        schema_info = {
            'schema_levels': [],
            'nested_schemas': {},
            'defaults': {},
            'raw_config': None
        }
        
        try:
            isaacgym_envs_path = get_isaac_gym_envs_path()
            task_cfg_path = isaacgym_envs_path / 'isaacgymenvs' / 'cfg' / 'task' / f'{task_name}.yaml'
            
            if task_cfg_path.exists():
                with open(task_cfg_path, 'r') as f:
                    task_config = yaml.safe_load(f)
                
                # Store raw config for later use
                schema_info['raw_config'] = task_config
                
                # Use utility functions to extract schema info
                schema_info['schema_levels'] = yaml_schema_utils.get_dnne_schema_levels(task_config)
                schema_info['defaults'] = yaml_schema_utils.get_schema_defaults(task_config)
                
                # Get nested schemas if present
                dnne_config = yaml_schema_utils.navigate_schema(task_config, ['env', 'dnne'])
                if not dnne_config:
                    dnne_config = yaml_schema_utils.navigate_schema(task_config, ['dnne'])
                
                if dnne_config and 'nested_schemas' in dnne_config:
                    schema_info['nested_schemas'] = dnne_config['nested_schemas']
                else:
                    # No DNNE config - check for basic env info
                    env_config = yaml_schema_utils.navigate_schema(task_config, ['env'])
                    if env_config:
                        if 'numObservations' in env_config or 'numActions' in env_config:
                            schema_info['nested_schemas'] = {
                                'numObservations': env_config.get('numObservations'),
                                'numActions': env_config.get('numActions')
                            }
        
        except Exception as e:
            # Unexpected errors should propagate
            raise RuntimeError(f"Failed to load schema for task {task_name}: {e}") from e
        
        return schema_info
    
    @classmethod
    def get_env_config(cls, node_id, node_data):
        """Query method to get environment configuration from this virtual node.
        
        This method is called by non-virtual nodes (like PPOAgent, IsaacGymSim) to retrieve
        environment configuration without directly accessing this node's widgets.
        """
        # Static widget indices:
        # 0: task
        # 1-3: dynamic levels
        # 4: dt
        # 5: num_envs
        # 6: seed
        # 7: seed_control
        # 8: headless
        # 9: graphics_device_id
        # 10: sim_device
        # 11: physics_engine
        # 12: multi_gpu
        # 13: enable_cameras
        # 14: force_render
        # 15: use_gpu_pipeline
        # 16: num_threads
        # 17: solver_type
        # 18: num_subscenes
        # 19: schema_display (read-only, ignore)
        
        # Build parameter specs with static indices
        param_specs = [
            {'name': 'task', 'widget_index': 0},
            # Dynamic widgets (may contain level values or "none")
            {'name': 'dynamic_1', 'widget_index': 1},
            {'name': 'dynamic_2', 'widget_index': 2},
            {'name': 'dynamic_3', 'widget_index': 3},
            # Fixed widgets
            {'name': 'dt', 'widget_index': 4},
            {'name': 'num_envs', 'widget_index': 5},
            {'name': 'seed', 'widget_index': 6},
            {'name': 'seed_control', 'widget_index': 7},
            {'name': 'headless', 'widget_index': 8},
            {'name': 'graphics_device_id', 'widget_index': 9},
            {'name': 'sim_device', 'widget_index': 10},
            {'name': 'physics_engine', 'widget_index': 11},
            {'name': 'multi_gpu', 'widget_index': 12},
            {'name': 'enable_cameras', 'widget_index': 13},
            {'name': 'force_render', 'widget_index': 14},
            {'name': 'use_gpu_pipeline', 'widget_index': 15},
            {'name': 'num_threads', 'widget_index': 16},
            {'name': 'solver_type', 'widget_index': 17},
            {'name': 'num_subscenes', 'widget_index': 18},
        ]
        
        # Get parameters using the helper
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Get task to load schema info
        task_name = params.get('task', 'Cartpole')
        schema_info = cls._load_task_schema(task_name)
        
        # Map dynamic widgets to their schema levels
        config = {
            'task': task_name,
            'dt': params.get('dt'),
            'num_envs': params.get('num_envs'),
            'seed': params.get('seed'),
            'seed_control': params.get('seed_control'),
            'headless': params.get('headless'),
            'graphics_device_id': params.get('graphics_device_id'),
            'sim_device': params.get('sim_device'),
            'physics_engine': params.get('physics_engine'),
            'multi_gpu': params.get('multi_gpu'),
            'enable_cameras': params.get('enable_cameras'),
            'force_render': params.get('force_render'),
            'use_gpu_pipeline': params.get('use_gpu_pipeline'),
            'num_threads': params.get('num_threads'),
            'solver_type': params.get('solver_type'),
            'num_subscenes': params.get('num_subscenes'),
        }
        
        # Add dynamic level values with their proper names
        for i, level in enumerate(schema_info.get('schema_levels', [])):
            if i < cls.MAX_DYNAMIC_LEVELS:
                value = params.get(f'dynamic_{i+1}')
                if value and value != 'none':
                    config[level] = value
        
        # Use utility functions to extract observation/action sizes from schema
        if schema_info.get('raw_config'):
            # Build level_values dict from config
            level_values = {}
            for level in schema_info.get('schema_levels', []):
                if level in config:
                    level_values[level] = config[level]
            
            # Extract sizes using utility function
            num_obs, num_acts = yaml_schema_utils.extract_observation_action_sizes(
                schema_info['raw_config'], level_values
            )
            
            if num_obs is not None:
                config['numObservations'] = num_obs
            if num_acts is not None:
                config['numActions'] = num_acts
            
            # Also get null action if available
            null_action = yaml_schema_utils.get_null_action(
                schema_info['raw_config'], level_values
            )
            if null_action is not None:
                config['nullAction'] = null_action
        
        # Add isaac_gym_envs_path for PPOAgent exporter
        config['isaac_gym_envs_path'] = str(get_isaac_gym_envs_path())
        
        return config