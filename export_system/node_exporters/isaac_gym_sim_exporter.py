#!/usr/bin/env python3
"""
Exporter for Isaac Gym Simulator node using queue-based template
"""

from ..graph_exporter import ExportableNode
from ..utils import export_utils

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
        
        # Get config from connected Isaac Gym Environment Config node using proper query method
        config_params = cls._get_env_config_via_query(node_id, all_nodes, all_links, node_registry)
        
        if not config_params:
            # Fail-fast: no config connected is an error
            raise ValueError(
                f"IsaacGymSim node {node_id} has no connected IsaacGymEnvs configuration node. "
                f"Please connect an IsaacGymEnvs node to the env_config input."
            )
        
        # Validate required parameters are present
        required_params = ['task', 'num_envs', 'seed', 'seed_control', 'headless',
                         'graphics_device_id', 'sim_device', 'physics_engine', 
                         'multi_gpu', 'enable_cameras']
        missing_params = [p for p in required_params if config_params.get(p) is None]
        if missing_params:
            raise ValueError(
                f"IsaacGymEnvs configuration missing required parameters: {missing_params}. "
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
        
        # Extract dynamic level values (subtask, controlType, etc.)
        subtask = config_params.get('subtask')
        control_type = config_params.get('controlType')
        
        # Extract schema values that IsaacGymEnvs should have provided
        num_observations = config_params.get('numObservations')
        num_actions = config_params.get('numActions')
        
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
        
        # Build dnne_cfg code snippet if needed
        dnne_cfg_code = ""
        if subtask or control_type or num_observations or num_actions:
            dnne_cfg_code = "\n            # Create dnne_cfg for environment-specific overrides\n"
            dnne_cfg_code += "            dnne_cfg = {}\n"
            
            if subtask:
                dnne_cfg_code += f"            \n"
                dnne_cfg_code += f"            # Add subtask for environments that use it (like FrankaDNNE)\n"
                dnne_cfg_code += f"            dnne_cfg.setdefault('env', {{}})[\'subtask\'] = \"{subtask}\"\n"
                dnne_cfg_code += f"            print(f\"[DEBUG IsaacGymSim] Setting subtask: {subtask}\")\n"
            
            if control_type:
                dnne_cfg_code += f"            \n"
                dnne_cfg_code += f"            # Add controlType for environments that use it (like FrankaDNNE)\n"
                dnne_cfg_code += f"            dnne_cfg.setdefault('env', {{}})[\'controlType\'] = \"{control_type}\"\n"
                dnne_cfg_code += f"            print(f\"[DEBUG IsaacGymSim] Setting controlType: {control_type}\")\n"
            
            if num_observations:
                dnne_cfg_code += f"            \n"
                dnne_cfg_code += f"            # Override numObservations from schema\n"
                dnne_cfg_code += f"            dnne_cfg.setdefault('env', {{}})[\'numObservations\'] = {num_observations}\n"
                dnne_cfg_code += f"            print(f\"[DEBUG IsaacGymSim] Setting numObservations: {num_observations}\")\n"
            
            if num_actions:
                dnne_cfg_code += f"            \n"
                dnne_cfg_code += f"            # Override numActions from schema\n"
                dnne_cfg_code += f"            dnne_cfg.setdefault('env', {{}})[\'numActions\'] = {num_actions}\n"
                dnne_cfg_code += f"            print(f\"[DEBUG IsaacGymSim] Setting numActions: {num_actions}\")\n"
        
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
            "DNNE_CFG_CODE": dnne_cfg_code,  # Conditional code snippet
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
                    
                    # Extract observation size and schema from env schema
                    if 'outputs' in env_schema and 'env' in env_schema['outputs']:
                        env_output = env_schema['outputs']['env']
                        if 'observation_size' in env_output:
                            # Update our observation output with the size
                            schema['outputs']['observation']['flattened_size'] = env_output['observation_size']
                            schema['outputs']['observation']['shape'] = [env_output['observation_size']]
                            del schema['outputs']['observation']['needs_resolution']  # Remove flag
                        
                        # Also propagate the observation schema if available
                        if 'observation_schema' in env_output:
                            schema['outputs']['observation']['observation_schema'] = env_output['observation_schema']
        
        return schema
    
    @classmethod
    def _get_env_config_via_query(cls, node_id, all_nodes, all_links, node_registry):
        """Get environment configuration from connected IsaacGymEnvs virtual node using query method.
        
        This method respects widget encapsulation by calling the IsaacGymEnvs exporter's
        query method instead of directly accessing its widgets.
        """
        # Export context should already be set by GraphExporter
        if not all_links or not all_nodes:
            raise RuntimeError(
                f"IsaacGymSim node {node_id}: Cannot get environment config - missing nodes or links data"
            )
            
        # Find the env_config input connection (slot 0)
        env_node_id = None
        for link in all_links:
            if len(link) >= 5 and str(link[3]) == str(node_id) and link[4] == 0:  # env_config is input 0
                env_node_id = str(link[1])
                break
        
        if not env_node_id:
            raise RuntimeError(
                f"IsaacGymSim node {node_id}: No environment configuration connected to env_config input. "
                f"Please connect an IsaacGymEnvs node."
            )
            
        # Find the node data
        env_node_data = export_utils.get_node_by_id(env_node_id)
        if not env_node_data:
            raise RuntimeError(
                f"IsaacGymSim node {node_id}: Connected environment node {env_node_id} not found in workflow"
            )
            
        # Check if it's an IsaacGymEnvs node
        node_type = env_node_data.get("class_type") or env_node_data.get("type")
        if node_type != "IsaacGymEnvs":
            raise RuntimeError(
                f"IsaacGymSim node {node_id}: Expected IsaacGymEnvs node connected to env_config, "
                f"but got {node_type} node instead"
            )
            
        # Get the IsaacGymEnvs exporter and call its query method
        env_exporter = export_utils.get_node_exporter("IsaacGymEnvs")
        if not env_exporter:
            raise ValueError(
                f"IsaacGymEnvs exporter not found. "
                f"This indicates a missing virtual node implementation."
            )
        
        # Call the query method to get configuration - let AttributeError propagate if method missing
        return env_exporter.get_env_config(env_node_id, env_node_data)
    
    @classmethod
    def _get_node_registry(cls):
        """Get the node registry for exporter lookups.
        
        This helper method builds a registry of node exporters.
        In production, this would be passed from GraphExporter.
        """
        # Import the exporters we need
        from .isaac_gym_envs_exporter import IsaacGymEnvsExporter
        
        return {
            'IsaacGymEnvs': IsaacGymEnvsExporter,
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