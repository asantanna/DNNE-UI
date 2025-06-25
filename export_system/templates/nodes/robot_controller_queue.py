# Template variables - replaced during export
template_vars = {
    "NODE_ID": "robot_1",
    "CLASS_NAME": "RobotControllerNode",
    "JOINT_LIMITS_MIN": -3.14,
    "JOINT_LIMITS_MAX": 3.14,
    "CONTROL_TYPE": "position",
    "NUM_JOINTS": 7
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Robot controller that converts actions to joint commands"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["action"])
        self.setup_outputs(["joint_commands", "status"])
        
        self.num_joints = {NUM_JOINTS}
        self.joint_limits = ({JOINT_LIMITS_MIN}, {JOINT_LIMITS_MAX})
        self.control_type = "{CONTROL_TYPE}"
        
    async def compute(self, action) -> Dict[str, Any]:
        # Convert action to joint commands
        # Placeholder implementation
        
        # Ensure action is the right size
        if hasattr(action, '__len__'):
            if len(action) != self.num_joints:
                self.logger.warning(f"Action size {{len(action)}} != num_joints {{self.num_joints}}")
        
        # Clip to joint limits
        joint_commands = []
        for i in range(self.num_joints):
            if hasattr(action, '__getitem__'):
                cmd = max(self.joint_limits[0], min(self.joint_limits[1], float(action[i])))
            else:
                cmd = 0.0  # Default position
            joint_commands.append(cmd)
        
        status = f"Sent {{self.control_type}} commands to {{self.num_joints}} joints"
        
        return {{
            "joint_commands": joint_commands,
            "status": status
        }}
