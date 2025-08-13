#!/usr/bin/env python3
"""
Exporter for Isaac Gym Simulator node using queue-based template
"""

from ..graph_exporter import ExportableNode

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
        # Debug: Check what connections and all_nodes look like
        import logging
        logging.info(f"[IsaacGymSim Export] Looking for env_config connection")
        logging.info(f"[IsaacGymSim Export] Node ID: {node_id}")
        logging.info(f"[IsaacGymSim Export] All nodes IDs: {[n.get('id') for n in all_nodes] if all_nodes else 'None'}")
        
        # Find connected IsaacGymEnvs node through links
        config_node = None
        if all_links:
            for link in all_links:
                # Link format: [link_id, from_node, from_slot, to_node, to_slot, type]
                if len(link) >= 5 and str(link[3]) == str(node_id) and link[4] == 0:  # env_config is input 0
                    source_node_id = str(link[1])
                    logging.info(f"[IsaacGymSim Export] Found connection from node {source_node_id}")
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
                {'name': 'num_envs', 'widget_index': 1},
                {'name': 'seed', 'widget_index': 2},
                {'name': 'seed_control', 'widget_index': 3},
                {'name': 'headless', 'widget_index': 4},
                {'name': 'graphics_device_id', 'widget_index': 5},
                {'name': 'sim_device', 'widget_index': 6},
                {'name': 'physics_engine', 'widget_index': 7},
                {'name': 'multi_gpu', 'widget_index': 8},
                {'name': 'enable_cameras', 'widget_index': 9},
                {'name': 'force_render', 'widget_index': 10},
                {'name': 'use_gpu_pipeline', 'widget_index': 11},
                {'name': 'num_threads', 'widget_index': 12},
                {'name': 'solver_type', 'widget_index': 13},
                {'name': 'num_subscenes', 'widget_index': 14},
            ]
            
            # Get parameters using the helper that checks both inputs and widgets_values
            config_params = cls.get_node_parameters_batch(config_node, param_specs)
            
            logging.info(f"[IsaacGymSim Export] Config node type: {config_node.get('type')}")
            logging.info(f"[IsaacGymSim Export] Extracted config_params: {config_params}")
            
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
            logging.info(f"[IsaacGymSim Export] Extracted task: {task}")
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
        camera_pos_str = params['camera_position'].strip()
        if camera_pos_str:
            camera_pos_list = [float(x.strip()) for x in camera_pos_str.split(',')]
            if len(camera_pos_list) != 3:
                raise ValueError(f"Camera position must have exactly 3 values (x,y,z), got {len(camera_pos_list)}")
        else:
            camera_pos_list = [1.2, 1.2, 1.0]  # Default
        
        # Parse camera target string to list
        camera_target_str = params['camera_target'].strip()
        if camera_target_str:
            camera_target_list = [float(x.strip()) for x in camera_target_str.split(',')]
            if len(camera_target_list) != 3:
                raise ValueError(f"Camera target must have exactly 3 values (x,y,z), got {len(camera_target_list)}")
        else:
            camera_target_list = [0.0, 0.0, 0.5]  # Default
        
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
        # Output schema depends on the environment
        # For now, return generic tensor output
        return {
            "outputs": {
                "observation": {
                    "type": "tensor",
                    "dtype": "float32"
                },
                "done": {
                    "type": "trigger"
                }
            }
        }
    
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