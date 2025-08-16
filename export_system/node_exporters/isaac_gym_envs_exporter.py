#!/usr/bin/env python3
"""
Exporter for IsaacGymEnvs node using queue-based template
"""

import yaml
import os
import sys
from pathlib import Path
from ..graph_exporter import ExportableNode

# Add parent directory to path to import dnne_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dnne_config import get_isaac_gym_envs_path

class IsaacGymEnvsExporter(ExportableNode):
    """Exporter for IsaacGymEnvs virtual node - provides environment configuration"""
    # IsaacGymEnvs is a virtual node - only provides configuration
    # Virtual status is handled by @dnne_node decorator
    
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
        # Extract task and subtask from node data
        param_specs = [
            {'name': 'task', 'widget_index': 0},
            {'name': 'subtask', 'widget_index': 1},
        ]
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        task_name = params.get('task', 'Cartpole')
        subtask = params.get('subtask', '')
        
        # Load task configuration to get observation/action sizes and schemas
        observation_size = None
        action_size = None
        observation_schema = None
        action_schema = None
        
        try:
            # Load YAML directly instead of using the processed config
            isaacgym_envs_path = get_isaac_gym_envs_path()
            task_cfg_path = isaacgym_envs_path / 'isaacgymenvs' / 'cfg' / 'task' / f'{task_name}.yaml'
            
            if task_cfg_path.exists():
                with open(task_cfg_path, 'r') as f:
                    task_config = yaml.safe_load(f)
                
                # Look for dnne section - check both root level and env level
                dnne_config = None
                if 'env' in task_config and 'dnne' in task_config['env']:
                    dnne_config = task_config['env']['dnne']
                elif 'dnne' in task_config:
                    dnne_config = task_config['dnne']
                
                if dnne_config:
                    if 'subtasks' in dnne_config:
                        # Get the specific subtask or default
                        if not subtask and 'defaultSubtask' in dnne_config:
                            subtask = dnne_config['defaultSubtask']
                        
                        if subtask in dnne_config['subtasks']:
                            subtask_config = dnne_config['subtasks'][subtask]
                            observation_size = subtask_config.get('numObservations')
                            action_size = subtask_config.get('numActions')
                            observation_schema = subtask_config.get('observationSchema')
                            action_schema = subtask_config.get('actionSchema')
                
                # Fallback to env section for sizes
                if observation_size is None and 'env' in task_config:
                    observation_size = task_config['env'].get('numObservations')
                if action_size is None and 'env' in task_config:
                    action_size = task_config['env'].get('numActions')
                
        except Exception as e:
            # If we can't load the config, sizes and schemas will remain None
            pass
        
        # Build schema
        schema = {
            "outputs": {
                "env": {
                    "type": "env_config",
                    "task": task_name,
                    "subtask": subtask
                }
            }
        }
        
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
    def get_env_config(cls, node_id, node_data):
        """Query method to get environment configuration from this virtual node.
        
        This method is called by non-virtual nodes (like PPOAgent, IsaacGymSim) to retrieve
        environment configuration without directly accessing this node's widgets.
        
        Args:
            node_id: The ID of this IsaacGymEnvs node
            node_data: The node data dictionary containing widget values
            
        Returns:
            Dictionary with environment configuration parameters
        """
        # Use parameter specs to extract values
        param_specs = [
            {'name': 'task', 'widget_index': 0},
            {'name': 'subtask', 'widget_index': 1},
            {'name': 'dt', 'widget_index': 2},
            {'name': 'num_envs', 'widget_index': 3},
            {'name': 'seed', 'widget_index': 4},
            {'name': 'seed_control', 'widget_index': 5},
            {'name': 'headless', 'widget_index': 6},
            {'name': 'graphics_device_id', 'widget_index': 7},
            {'name': 'sim_device', 'widget_index': 8},
            {'name': 'physics_engine', 'widget_index': 9},
            {'name': 'multi_gpu', 'widget_index': 10},
            {'name': 'enable_cameras', 'widget_index': 11},
            {'name': 'force_render', 'widget_index': 12},
            {'name': 'use_gpu_pipeline', 'widget_index': 13},
            {'name': 'num_threads', 'widget_index': 14},
            {'name': 'solver_type', 'widget_index': 15},
            {'name': 'num_subscenes', 'widget_index': 16},
        ]
        
        # Get parameters using the helper
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Validate required parameters are present
        required_params = ['task', 'num_envs', 'seed', 'seed_control', 'headless',
                          'graphics_device_id', 'sim_device', 'physics_engine', 
                          'multi_gpu', 'enable_cameras']
        missing_params = [p for p in required_params if params.get(p) is None]
        if missing_params:
            raise ValueError(
                f"IsaacGymEnvs node {node_id} missing required parameters: {missing_params}. "
                f"This may indicate the UI is not sending widget values correctly."
            )
        
        # Add isaac_gym_envs_path
        params['isaac_gym_envs_path'] = str(get_isaac_gym_envs_path())
        
        return params


# Registration function