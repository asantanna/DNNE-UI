#!/usr/bin/env python3
"""
Exporter for IsaacGymEnvs node using queue-based template
"""

import yaml
import os
from pathlib import Path
from ..graph_exporter import ExportableNode

class IsaacGymEnvsExporter(ExportableNode):
    """Exporter for IsaacGymEnvs virtual node - provides environment configuration"""
    
    @classmethod
    def is_virtual(cls):
        """IsaacGymEnvs is a virtual node - only provides configuration"""
        return True
    
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
        
        # Load task configuration to get observation/action sizes
        observation_size = None
        action_size = None
        
        try:
            # Load YAML directly instead of using the processed config
            isaacgym_envs_path = Path('/home/asantanna/DNNE/DNNE-LINUX-SUPPORT/IsaacGymEnvs')
            task_cfg_path = isaacgym_envs_path / 'isaacgymenvs' / 'cfg' / 'task' / f'{task_name}.yaml'
            
            if task_cfg_path.exists():
                with open(task_cfg_path, 'r') as f:
                    task_config = yaml.safe_load(f)
                
                # Look for dnne section at the root level
                if 'dnne' in task_config:
                    dnne_config = task_config['dnne']
                    
                    if 'subtasks' in dnne_config:
                        # Get the specific subtask or default
                        if not subtask and 'defaultSubtask' in dnne_config:
                            subtask = dnne_config['defaultSubtask']
                        
                        if subtask in dnne_config['subtasks']:
                            subtask_config = dnne_config['subtasks'][subtask]
                            observation_size = subtask_config.get('numObservations')
                            action_size = subtask_config.get('numActions')
                
                # Fallback to env section for sizes
                if observation_size is None and 'env' in task_config:
                    observation_size = task_config['env'].get('numObservations')
                if action_size is None and 'env' in task_config:
                    action_size = task_config['env'].get('numActions')
                
        except Exception as e:
            # If we can't load the config, sizes will remain None
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
        
        return schema
    
    @classmethod
    def _extract_env_config(cls, target_node_id, all_nodes, all_links):
        """Extract environment configuration from connected IsaacGymEnvs node.
        
        This method is used by other exporters (PPOAgent, IsaacGymSim) to get
        environment configuration from a connected IsaacGymEnvs node.
        """
        if not all_links or not all_nodes:
            return None
            
        # Find the env input connection to the target node
        env_node_id = None
        for link in all_links:
            if len(link) >= 5:
                to_node, to_slot = str(link[3]), link[4]
                if to_node == target_node_id and to_slot == 0:  # env input is typically slot 0
                    env_node_id = str(link[1])
                    break
        
        if not env_node_id:
            return None
            
        # Find the node data
        env_node_data = None
        for node in all_nodes:
            if str(node["id"]) == env_node_id:
                env_node_data = node
                break
                
        if not env_node_data:
            return None
            
        # Check if it's an IsaacGymEnvs node
        node_type = env_node_data.get("class_type") or env_node_data.get("type")
        if node_type != "IsaacGymEnvs":
            return None
            
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
        params = cls.get_node_parameters_batch(env_node_data, param_specs)
        
        # Validate required parameters are present
        required_params = ['task', 'num_envs', 'seed', 'seed_control', 'headless',
                          'graphics_device_id', 'sim_device', 'physics_engine', 
                          'multi_gpu', 'enable_cameras']
        missing_params = [p for p in required_params if params.get(p) is None]
        if missing_params:
            raise ValueError(
                f"IsaacGymEnvs node {env_node_id} missing required parameters: {missing_params}. "
                f"This may indicate the UI is not sending widget values correctly."
            )
        
        # Add isaac_gym_envs_path
        params['isaac_gym_envs_path'] = '/home/asantanna/DNNE/DNNE-LINUX-SUPPORT/IsaacGymEnvs'
        
        return params


# Registration function