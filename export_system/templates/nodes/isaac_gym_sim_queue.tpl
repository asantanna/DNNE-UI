# Template variables - replaced during export
template_vars = {
    "NODE_ID": "isaac_gym_sim_1",
    "CLASS_NAME": "IsaacGymSimNode",
    "RESET_WHEN_DONE": True,
    "RENDER": False,
    "NULL_ACTION": [0.0],
    "TASK": "Cartpole",
    "NUM_ENVS": 64,
    "SEED": 42,
    "HEADLESS": True,
    "SIM_DEVICE": "cuda:0",
    "PHYSICS_ENGINE": "physx",
    "GRAPHICS_DEVICE_ID": 0,
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Isaac Gym Simulator Interface - Queue-based environment wrapper"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["action"])  # Config is embedded, reset handled in compute
        self.setup_outputs(["observation", "done"])
        
        # Create input queue for optional reset input
        from asyncio import Queue
        self.input_queues["reset"] = Queue(maxsize=1)  # Trigger signal - size 1
        
        self.reset_when_done = {RESET_WHEN_DONE}
        # Check for visual mode from Global
        from framework.globals import Global as g
        self.render = g.visual_mode if hasattr(g, 'visual_mode') else {RENDER}
        self.null_action = {NULL_ACTION}
        self.env = None
        self.device = None
        self.obs_space = None
        self.act_space = None
        self.num_envs = 1  # Force single environment for DNNE
        
    async def initialize(self):
        """Initialize Isaac Gym environment from config"""
        print(f"[DEBUG IsaacGymSim] Starting initialization...")
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
            
            # Import isaacgymenvs (already imported at top of runner.py)
            print(f"[DEBUG IsaacGymSim] Importing isaacgymenvs...")
            from isaacgymenvs import make
            
            print(f"[DEBUG IsaacGymSim] Creating env_config with task={{config['task']}}, render={{self.render}}")
            
            # Override num_envs to 1 for DNNE compatibility
            env_config = {{
                "task": config.get("task", "{TASK}"),
                "num_envs": 1,  # Force single environment
                "seed": config.get("seed", {SEED}),
                "sim_device": config.get("sim_device", "{SIM_DEVICE}"),
                "rl_device": config.get("sim_device", "{SIM_DEVICE}"),
                "graphics_device_id": config.get("graphics_device_id", {GRAPHICS_DEVICE_ID}),
                "headless": config.get("headless", {HEADLESS}) and not self.render,
                "force_render": self.render,
                "multi_gpu": False,  # Not supported with single env
                "virtual_screen_capture": False,
                "enable_cameras": config.get("enable_cameras", False),
            }}
            
            # Create environment
            print(f"[DEBUG IsaacGymSim] Calling make() with config: {{env_config}}")
            self.env = make(**env_config)
            print(f"[DEBUG IsaacGymSim] Environment created successfully!")
            
            # Get device
            self.device = torch.device(config.get("sim_device", "{SIM_DEVICE}"))
            
            # Get observation and action spaces
            self.obs_space = self.env.observation_space
            self.act_space = self.env.action_space
            
            # Get initial observation
            obs = self.env.reset()
            if isinstance(obs, dict):
                obs = obs["obs"]
            
            # If we have a null action, step once to get proper initial observation
            if self.null_action:
                # Validate null action is provided
                if not self.null_action:
                    raise ValueError(f"No null action provided for task {{config['task']}}. "
                                   "Please specify null_action parameter or add nullAction to YAML config.")
                
                # Create null action tensor
                null_action_tensor = torch.tensor(self.null_action, device=self.device, dtype=torch.float32)
                if null_action_tensor.dim() == 1:
                    null_action_tensor = null_action_tensor.unsqueeze(0)
                
                # Step with null action to get initial observation
                obs, _, _, _ = self.env.step(null_action_tensor)
                if isinstance(obs, dict):
                    obs = obs["obs"]
                
                self.node_logger.info(f"Stepped with null action: {{self.null_action}}")
            
            # Send initial observation
            await self.send_output("observation", obs)
            
            self.node_logger.info(f"Initialized {{config['task']}} environment on {{self.device}}")
            
        except Exception as e:
            self.node_logger.error(f"Failed to initialize environment: {{e}}")
            raise
    
    async def compute(self, **kwargs) -> Dict[str, Any]:
        """Step the environment or handle reset"""
        print(f"[DEBUG IsaacGymSim] compute() called with kwargs keys: {{kwargs.keys()}}")
        action = kwargs.get('action')
        reset = kwargs.get('reset')
        
        # Initialize on first call
        if self.env is None:
            print(f"[DEBUG IsaacGymSim] env is None, calling initialize()...")
            await self.initialize()
            return {{}}  # Initialization sends initial observation
        
        # Handle manual reset
        if reset is not None:
            obs = self.env.reset()
            if isinstance(obs, dict):
                obs = obs["obs"]
            
            # Step with null action after reset if available
            if self.null_action:
                null_action_tensor = torch.tensor(self.null_action, device=self.device, dtype=torch.float32)
                if null_action_tensor.dim() == 1:
                    null_action_tensor = null_action_tensor.unsqueeze(0)
                obs, _, _, _ = self.env.step(null_action_tensor)
                if isinstance(obs, dict):
                    obs = obs["obs"]
            
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
            
            # Prepare outputs
            outputs = {{"observation": obs}}
            
            # Handle done signal
            if done.any():
                outputs["done"] = True  # Send trigger
                
                # Auto-reset if configured
                if self.reset_when_done:
                    obs = self.env.reset()
                    if isinstance(obs, dict):
                        obs = obs["obs"]
                    
                    # Step with null action after reset if available
                    if self.null_action:
                        null_action_tensor = torch.tensor(self.null_action, device=self.device, dtype=torch.float32)
                        if null_action_tensor.dim() == 1:
                            null_action_tensor = null_action_tensor.unsqueeze(0)
                        obs, _, _, _ = self.env.step(null_action_tensor)
                        if isinstance(obs, dict):
                            obs = obs["obs"]
                    
                    # Send new observation after reset
                    await self.send_output("observation", obs)
            
            return outputs
        
        return {{}}
    
    async def cleanup(self):
        """Clean up environment resources"""
        if self.env is not None:
            try:
                self.env.close()
            except:
                pass