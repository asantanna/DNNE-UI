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
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Get config from connected Isaac Gym Environment Config node
        config_connection = cls.find_input_connection(node_data, connections, "env_config")
        if config_connection:
            source_node_id = config_connection['from_node']
            config_node = all_nodes.get(source_node_id, {})
            
            # Extract config values from the config node's widgets
            config_widgets = config_node.get('widgets_values', [])
            
            # Map widget indices to config parameters
            # Based on INPUT_TYPES order in isaac_gym_envs_visnode.py
            task = config_widgets[0] if len(config_widgets) > 0 else "Cartpole"
            num_envs = config_widgets[1] if len(config_widgets) > 1 else 64
            seed = config_widgets[2] if len(config_widgets) > 2 else 42
            seed_control = config_widgets[3] if len(config_widgets) > 3 else "fixed"
            headless = config_widgets[4] if len(config_widgets) > 4 else True
            graphics_device_id = config_widgets[5] if len(config_widgets) > 5 else 0
            sim_device = config_widgets[6] if len(config_widgets) > 6 else "cuda:0"
            physics_engine = config_widgets[7] if len(config_widgets) > 7 else "physx"
            multi_gpu = config_widgets[8] if len(config_widgets) > 8 else False
            enable_cameras = config_widgets[9] if len(config_widgets) > 9 else False
        else:
            # Default values if no config connected
            task = "Cartpole"
            num_envs = 64
            seed = 42
            headless = True
            graphics_device_id = 0
            sim_device = "cuda:0"
            physics_engine = "physx"
        
        # Parse null action string to list
        null_action_str = params['null_action'].strip()
        if null_action_str:
            null_action_list = [float(x.strip()) for x in null_action_str.split(',')]
        else:
            null_action_list = []
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "IsaacGymSimNode",
            "RESET_WHEN_DONE": params['reset_when_done'],
            "RENDER": params['render'],
            "NULL_ACTION": null_action_list,
            "TASK": task,
            "NUM_ENVS": num_envs,
            "SEED": seed,
            "HEADLESS": headless,
            "SIM_DEVICE": f'"{sim_device}"',  # Add quotes for string
            "PHYSICS_ENGINE": f'"{physics_engine}"',  # Add quotes for string
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