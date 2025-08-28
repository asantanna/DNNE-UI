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
    "DNNE_CFG_CODE": "",  # Conditional code for dnne_cfg
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Isaac Gym Simulator Interface - Queue-based environment wrapper"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=[])  # Override to no required inputs - we bootstrap with null_action
        self.setup_outputs(["observation", "done"])
        
        # Manually create input queues for action and reset
        from asyncio import Queue
        self.input_queues["action"] = Queue(maxsize=2)  # Action input
        self.input_queues["reset"] = Queue(maxsize=1)  # Trigger signal - size 1
        
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
        """Custom run method: bootstrap with null_action, then process actions normally"""
        import time
        self.running = True
        self.node_logger.info(f"Starting IsaacGymSim node {{self.node_id}}")
        
        try:
            # Bootstrap environment with null_action
            self.node_logger.info("Bootstrapping environment with null_action...")
            await self.initialize()
            # Initial observation already sent by initialize()
            
            # Main loop - wait for action OR reset
            while self.running:
                # Log waiting for inputs (with deadlock monitoring)
                if g.deadlock_debug:
                    from framework.deadlock_utils import log_queue_get_wait, log_queue_get_success
                    import time
                    # Log that we're waiting for either action or reset
                    log_queue_get_wait(self.node_id, "action_or_reset")
                    wait_start = time.time()
                
                # Create tasks for both inputs
                action_task = asyncio.create_task(self.input_queues["action"].get(), name="action")
                reset_task = asyncio.create_task(self.input_queues["reset"].get(), name="reset")
                
                # Wait for either action or reset
                done, pending = await asyncio.wait(
                    [action_task, reset_task], 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel the other task
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                # Process the completed input
                completed_task = list(done)[0]
                input_name = completed_task.get_name()
                
                # Log successful receipt (with deadlock monitoring)
                if g.deadlock_debug:
                    log_queue_get_success(self.node_id, input_name, time.time() - wait_start)
                
                if input_name == "action":
                    action = completed_task.result()
                    outputs = await self.compute(action=action)
                elif input_name == "reset":
                    reset = completed_task.result()
                    outputs = await self.compute(reset=reset)
                
                # Send outputs
                for output_name, value in outputs.items():
                    await self.send_output(output_name, value)
                    
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
            
            # Import isaacgymenvs (already imported at top of runner.py)
            from isaacgymenvs import make
            
            {DNNE_CFG_CODE}
            # Override num_envs to 1 for DNNE compatibility
            # Only include parameters that make() actually accepts
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
                # Note: enable_cameras is not a parameter for make(), it's handled via cfg
            }}
            
            # Add dnne_cfg if it was created
            if 'dnne_cfg' in locals() and dnne_cfg:
                env_config['dnne_cfg'] = dnne_cfg
            
            # Create environment
            self.env = make(**env_config)
            
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
                    f"FAIL-FAST: Task '{{task_name}}' requires null_action but none provided!\n"
                    f"The null_action must be extracted from the task's YAML schema.\n"
                    f"This is a critical failure in the export pipeline:\n"
                    f"1. IsaacGymEnvs node should extract nullAction from the selected schema\n"
                    f"2. Export system should validate null_action is present\n"
                    f"3. This node should fail immediately without null_action\n"
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
            
            self.node_logger.info(f"Bootstrapped with null action: {{self.null_action}}")
            
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
            
            # Prepare outputs
            outputs = {{"observation": obs}}
            
            # Handle done signal (any environment done triggers reset)
            if done.any():
                outputs["done"] = True  # Send trigger
                
                # Auto-reset if configured
                if self.reset_when_done:
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
                    
                    # Enable gradients for training mode (after reset)
                    if not g.inference_mode:
                        obs = obs.detach().requires_grad_(True)
                    
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