#!/usr/bin/env python3
"""
Exporter for Isaac Gym Simulator node using queue-based template
"""

from ..graph_exporter import ExportableNode
from .isaac_gym_envs_exporter import IsaacGymEnvsExporter

class IsaacGymSimExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/isaac_gym_sim_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Get parameters from widgets - NO DEFAULTS, fail-fast!
        param_specs = [
            {'name': 'reset_when_done', 'widget_index': 0},
            {'name': 'render', 'widget_index': 1},
            {'name': 'null_action', 'widget_index': 2},
            {'name': 'camera_position', 'widget_index': 3},
            {'name': 'camera_target', 'widget_index': 4},
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Validate required parameters are present
        required_params = ['reset_when_done', 'render', 'null_action', 'camera_position', 'camera_target']
        missing_params = [p for p in required_params if params.get(p) is None]
        if missing_params:
            raise ValueError(
                f"IsaacGymSim node {node_id} missing required parameters: {missing_params}. "
                f"This may indicate the UI is not sending widget values correctly."
            )
        
        # Get config from connected Isaac Gym Environment Config node
        
        # Find connected IsaacGymEnvs node through links
        config_node = None
        if all_links:
            for link in all_links:
                # Link format: [link_id, from_node, from_slot, to_node, to_slot, type]
                if len(link) >= 5 and str(link[3]) == str(node_id) and link[4] == 0:  # env_config is input 0
                    source_node_id = str(link[1])
                    # Find the source node
                    for node in all_nodes:
                        if str(node.get('id')) == source_node_id:
                            config_node = node
                            break
                    break
        
        if config_node:
            
            # Use parameter specs to extract values from either inputs dict or widgets_values
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
            
            # Get parameters using the helper that checks both inputs and widgets_values
            config_params = cls.get_node_parameters_batch(config_node, param_specs)
            
            # Validate required parameters are present
            required_params = ['task', 'num_envs', 'seed', 'seed_control', 'headless',
                             'graphics_device_id', 'sim_device', 'physics_engine', 
                             'multi_gpu', 'enable_cameras']
            missing_params = [p for p in required_params if config_params.get(p) is None]
            if missing_params:
                raise ValueError(
                    f"IsaacGymEnvs node missing required parameters: {missing_params}. "
                    f"This may indicate the UI is not sending widget values correctly."
                )
            
            # Extract individual values
            task = config_params['task']
            num_envs = config_params['num_envs']
            seed = config_params['seed']
            seed_control = config_params['seed_control']
            headless = config_params['headless']
            graphics_device_id = config_params['graphics_device_id']
            sim_device = config_params['sim_device']
            physics_engine = config_params['physics_engine']
            multi_gpu = config_params['multi_gpu']
            enable_cameras = config_params['enable_cameras']
        else:
            # Fail-fast: no config connected is an error
            raise ValueError(
                f"IsaacGymSim node {node_id} has no connected IsaacGymEnvs configuration node. "
                f"Please connect an IsaacGymEnvs node to the env_config input."
            )
        
        # Parse null action string to list
        null_action_str = params['null_action'].strip()
        if null_action_str:
            null_action_list = [float(x.strip()) for x in null_action_str.split(',')]
        else:
            null_action_list = []
        
        # Parse camera position string to list
        # FAIL-FAST: Field must exist, but empty value gets default
        if 'camera_position' not in params:
            raise ValueError(
                f"IsaacGymSim node {node_id} missing camera_position field. "
                f"The UI must provide this field (even if empty for default)."
            )
        camera_pos_str = params['camera_position'].strip()
        if camera_pos_str:
            camera_pos_list = [float(x.strip()) for x in camera_pos_str.split(',')]
            if len(camera_pos_list) != 3:
                raise ValueError(f"Camera position must have exactly 3 values (x,y,z), got {len(camera_pos_list)}")
        else:
            camera_pos_list = [1.2, 1.2, 1.0]  # Default when field exists but is empty
        
        # Parse camera target string to list  
        # FAIL-FAST: Field must exist, but empty value gets default
        if 'camera_target' not in params:
            raise ValueError(
                f"IsaacGymSim node {node_id} missing camera_target field. "
                f"The UI must provide this field (even if empty for default)."
            )
        camera_target_str = params['camera_target'].strip()
        if camera_target_str:
            camera_target_list = [float(x.strip()) for x in camera_target_str.split(',')]
            if len(camera_target_list) != 3:
                raise ValueError(f"Camera target must have exactly 3 values (x,y,z), got {len(camera_target_list)}")
        else:
            camera_target_list = [0.0, 0.0, 0.5]  # Default when field exists but is empty
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "IsaacGymSimNode",
            "RESET_WHEN_DONE": params['reset_when_done'],
            "RENDER": params['render'],
            "NULL_ACTION": null_action_list,
            "CAMERA_POSITION": camera_pos_list,
            "CAMERA_TARGET": camera_target_list,
            "TASK": task,
            "NUM_ENVS": num_envs,
            "SEED": seed,
            "HEADLESS": headless,
            "SIM_DEVICE": sim_device,  # String value, no extra quotes needed
            "PHYSICS_ENGINE": physics_engine,  # String value, no extra quotes needed
            "GRAPHICS_DEVICE_ID": graphics_device_id,
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import numpy as np",
            "from typing import Dict, Any, Optional",
            # isaacgymenvs imported in runner.py already
        ]
    
    @classmethod
    def get_output_names(cls):
        return ["observation", "done"]
    
    @classmethod
    def get_input_names(cls):
        return ["env_config", "action", "reset"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        """Return output schema - will be resolved with env config during export"""
        # Basic schema - observation size will be resolved later
        return {
            "outputs": {
                "observation": {
                    "type": "tensor",
                    "dtype": "float32",
                    "needs_resolution": True  # Flag for resolution during export
                },
                "done": {
                    "type": "trigger"
                }
            }
        }
    
    @classmethod
    def get_output_schema(cls, node_data, connections=None, node_registry=None, 
                         all_nodes=None, all_links=None):
        """Get output schema by querying connected env config"""
        # Start with initial schema
        schema = cls.get_initial_output_schema(node_data)
        
        # Try to get env config from connected node
        if all_nodes and all_links:
            node_id = str(node_data.get('id', ''))
            
            # Find connected env_config node
            env_config_node = None
            for link in all_links:
                if len(link) >= 5 and str(link[3]) == node_id and link[4] == 0:  # env_config is input 0
                    source_node_id = str(link[1])
                    # Find the source node
                    for node in all_nodes:
                        if str(node.get('id')) == source_node_id:
                            env_config_node = node
                            break
                    break
            
            if env_config_node:
                # Get the env config node's schema
                node_type = env_config_node.get('class_type') or env_config_node.get('type')
                if node_type == 'IsaacGymEnvs' and node_registry and 'IsaacGymEnvs' in node_registry:
                    env_exporter = node_registry['IsaacGymEnvs']
                    env_schema = env_exporter.get_output_schema(env_config_node, connections, 
                                                                node_registry, all_nodes, all_links)
                    
                    # Extract observation size from env schema
                    if 'outputs' in env_schema and 'env' in env_schema['outputs']:
                        env_output = env_schema['outputs']['env']
                        if 'observation_size' in env_output:
                            # Update our observation output with the size
                            schema['outputs']['observation']['flattened_size'] = env_output['observation_size']
                            schema['outputs']['observation']['shape'] = [env_output['observation_size']]
                            del schema['outputs']['observation']['needs_resolution']  # Remove flag
        
        return schema
    
    @classmethod
    def find_input_connection(cls, node_data, connections, input_name):
        """Find connection to a specific input"""
        node_id = str(node_data.get('id', ''))
        
        # Handle both string and dict connection formats
        for conn in connections:
            # Skip if conn is just a string
            if isinstance(conn, str):
                continue
                
            # Handle dictionary connections
            if isinstance(conn, dict) and str(conn.get('to')) == node_id:
                # Check if this connection goes to the right input
                # In DNNE, inputs are indexed, we need to map names to indices
                input_names = cls.get_input_names()
                if input_name in input_names:
                    input_index = input_names.index(input_name)
                    if conn.get('to_socket', 0) == input_index:
                        return conn
        return None