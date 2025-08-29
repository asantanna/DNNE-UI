# Template variables
template_vars = {
    "NODE_ID": "isaacgymsimnode_1",
    "CLASS_NAME": "IsaacGymSimNode",
    "TASK": "FrankaDNNE",
    "NUM_ENVS": 1,
    "RENDER": False,
    "RESET_WHEN_DONE": True,
    "NULL_ACTION": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "CAMERA_POSITION": [3.0, 3.0, 3.0],
    "CAMERA_TARGET": [0.0, 0.0, 0.5],
    "SEED": 42,
    "HEADLESS": True,
    "SIM_DEVICE": "cuda:0",
    "PHYSICS_ENGINE": "physx",
    "GRAPHICS_DEVICE_ID": 0,
    "DNNE_CFG_CODE": "",  # Conditional code for dnne_cfg
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Isaac Gym Simulator Interface - Queue-based environment wrapper"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        # Use standard setup with both as optional since we bootstrap first
        self.setup_inputs(required=[], optional=["action", "reset"])
        self.setup_outputs(["observation", "done"])
        
        self.reset_when_done = {RESET_WHEN_DONE}
        # Check for visual mode from Global
        from framework.globals import Global as g
        self.render = g.visual_mode if hasattr(g, 'visual_mode') else {RENDER}
        self.null_action = {NULL_ACTION}
        self.camera_position = {CAMERA_POSITION}
        self.camera_target = {CAMERA_TARGET}
        self.env = None
        self.device = None
        self.obs_space = None
        self.act_space = None
        self.num_envs = 1  # Force single environment for DNNE
        
    async def run(self):
        """Override run to bootstrap environment before starting main loop"""
        self.running = True
        self.node_logger.info(f"Starting IsaacGymSim node {{self.node_id}}")
        
        try:
            # Bootstrap environment with null_action
            self.node_logger.info("Bootstrapping environment with null_action...")
            await self.initialize()
            # Initial observation already sent by initialize()
            
            # Now run the standard QueueNode loop with MultiWaiter
            await super().run()
            
        except asyncio.CancelledError:
            self.node_logger.info(f"IsaacGymSim Node {{self.node_id}} cancelled")
            raise
        finally:
            self.running = False
        
    async def initialize(self):
        """Initialize Isaac Gym environment from config"""
        try:
            # Config from IsaacGymEnvs virtual node (embedded during export)
            config = {{
                "task": "{TASK}",
                "num_envs": {NUM_ENVS},
                "seed": {SEED},
                "sim_device": "{SIM_DEVICE}",
                "headless": {HEADLESS},  # Will be overridden by visual flag
                "graphics_device_id": {GRAPHICS_DEVICE_ID},
                "enable_cameras": False
            }}

            if config["num_envs"] != 1:
                raise ValueError(f"Only num_envs=1 supported, got {config['num_envs']}")
            
            # Import isaacgymenvs (already imported at top of runner.py)
            from isaacgymenvs import make
            
{DNNE_CFG_CODE}
            
            # Apply visual mode
            if self.render:
                config["headless"] = False
            else:
                config["headless"] = True
            
            # Create environment with config + overrides
            self.env = make(
                task=config["task"],
                num_envs=config["num_envs"], 
                sim_device=config["sim_device"],
                rl_device=config.get("sim_device"),
                graphics_device_id=config["graphics_device_id"],
                headless=config["headless"],
                seed=config["seed"],
                dnne_cfg=dnne_cfg
            )
            
            # Set camera position from widget configuration
            if hasattr(self.env, 'viewer') and self.env.viewer is not None and self.camera_position and self.camera_target:
                # Use configured camera position and target
                from isaacgym import gymapi
                cam_pos = gymapi.Vec3(self.camera_position[0], self.camera_position[1], self.camera_position[2])
                cam_target = gymapi.Vec3(self.camera_target[0], self.camera_target[1], self.camera_target[2])
                self.env.gym.viewer_camera_look_at(
                    self.env.viewer, None, cam_pos, cam_target)
            
            # Get device
            self.device = torch.device(config.get("sim_device", "{SIM_DEVICE}"))
            
            # Get observation and action spaces
            self.obs_space = self.env.observation_space
            self.act_space = self.env.action_space
            
            # Get initial observation
            obs = self.env.reset()
            if isinstance(obs, dict):
                obs = obs["obs"]
            
            # ALL Isaac Gym tasks REQUIRE null_action to bootstrap the action-observation loop
            task_name = config.get('task', 'unknown')
            if not self.null_action or self.null_action == [] or self.null_action == "":
                raise ValueError(
                    f"FAIL-FAST: Task '{{task_name}}' requires null_action but none provided!\\n"
                    f"The null_action must be extracted from the task's YAML schema.\\n"
                    f"This is a critical failure in the export pipeline:\\n"
                    f"1. IsaacGymEnvs node should extract nullAction from the selected schema\\n"
                    f"2. Export system should validate null_action is present\\n"
                    f"3. This node should fail immediately without null_action\\n"
                    f"Check the task YAML for the nullAction field in the selected schema."
                )
            
            # Create null action tensor for bootstrapping
            null_action_tensor = torch.tensor(self.null_action, device=self.device, dtype=torch.float32)
            if null_action_tensor.dim() == 1:
                null_action_tensor = null_action_tensor.unsqueeze(0)
            
            # Step with null action to get proper initial observation
            obs, _, _, _ = self.env.step(null_action_tensor)
            if isinstance(obs, dict):
                obs = obs["obs"]
            
            # Enable gradients for training mode (Sim is a data source like DataStreamer)
            from framework.globals import Global as g
            if not g.inference_mode:
                obs = obs.detach().requires_grad_(True)
            
            # Send initial observation 
            await self.send_output("observation", obs)
            
            self.node_logger.info(f"Initialized {{config['task']}} environment on {{self.device}}")
            
        except Exception as e:
            self.node_logger.error(f"Failed to initialize environment: {{e}}")
            raise
    
    async def compute(self, **kwargs) -> Dict[str, Any]:
        """Step the environment or handle reset"""
        action = kwargs.get('action')
        reset = kwargs.get('reset')
        
        # Handle manual reset
        if reset is not None:
            obs = self.env.reset()
            if isinstance(obs, dict):
                obs = obs["obs"]
            
            # Step with null action after reset (required for all tasks)
            null_action_tensor = torch.tensor(self.null_action, device=self.device, dtype=torch.float32)
            if null_action_tensor.dim() == 1:
                null_action_tensor = null_action_tensor.unsqueeze(0)
            obs, _, _, _ = self.env.step(null_action_tensor)
            if isinstance(obs, dict):
                obs = obs["obs"]
            
            # Enable gradients for training mode
            from framework.globals import Global as g
            if not g.inference_mode:
                obs = obs.detach().requires_grad_(True)
            
            return {{"observation": obs}}
        
        # Execute action if provided
        if action is not None:
            # Ensure action is on correct device
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, device=self.device, dtype=torch.float32)
            else:
                action = action.to(self.device)
            
            # Ensure action has correct shape (add batch dim if needed)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            
            # Step environment
            obs, reward, done, info = self.env.step(action)
            
            # Extract observation if dict
            if isinstance(obs, dict):
                obs = obs["obs"]
            
            # Enable gradients for training mode
            from framework.globals import Global as g
            if not g.inference_mode:
                obs = obs.detach().requires_grad_(True)
            
            # Auto-reset if done and configured
            if done.any() and self.reset_when_done:
                obs = self.env.reset()
                if isinstance(obs, dict):
                    obs = obs["obs"]
                # Step with null action after reset
                null_action_tensor = torch.tensor(self.null_action, device=self.device, dtype=torch.float32)
                if null_action_tensor.dim() == 1:
                    null_action_tensor = null_action_tensor.unsqueeze(0)
                obs, _, _, _ = self.env.step(null_action_tensor)
                if isinstance(obs, dict):
                    obs = obs["obs"]
                if not g.inference_mode:
                    obs = obs.detach().requires_grad_(True)
            
            # Prepare outputs
            outputs = {{"observation": obs}}
            
            # Send done signal if episode ended
            if done.any():
                outputs["done"] = True
            
            return outputs
        
        # This shouldn't happen with required=["action"]
        return {{}}