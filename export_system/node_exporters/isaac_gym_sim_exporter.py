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
        # Get parameters from widgets
        param_specs = [
            {'name': 'reset_when_done', 'widget_index': 0, 'default': True},
            {'name': 'render', 'widget_index': 1, 'default': False},
            {'name': 'null_action', 'widget_index': 2, 'default': ""},
            {'name': 'camera_position', 'widget_index': 3, 'default': "1.2, 1.2, 1.0"},
            {'name': 'camera_target', 'widget_index': 4, 'default': "0.0, 0.0, 0.5"},
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
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
            
            # Extract config values from the config node's widgets
            config_widgets = config_node.get('widgets_values', [])
            logging.info(f"[IsaacGymSim Export] Config node type: {config_node.get('type')}")
            logging.info(f"[IsaacGymSim Export] Config widgets: {config_widgets}")
            
            # Map widget indices to config parameters
            # Based on INPUT_TYPES order in isaac_gym_envs_visnode.py
            # Fail-fast: ensure we have all required values
            if len(config_widgets) < 10:
                raise ValueError(
                    f"IsaacGymEnvs node has insufficient widget values ({len(config_widgets)}). "
                    f"Expected at least 10 values. This may indicate a corrupted workflow."
                )
            
            task = config_widgets[0]
            logging.info(f"[IsaacGymSim Export] Extracted task: {task}")
            num_envs = config_widgets[1]
            seed = config_widgets[2]
            seed_control = config_widgets[3]
            headless = config_widgets[4]
            graphics_device_id = config_widgets[5]
            sim_device = config_widgets[6]
            physics_engine = config_widgets[7]
            multi_gpu = config_widgets[8]
            enable_cameras = config_widgets[9]
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