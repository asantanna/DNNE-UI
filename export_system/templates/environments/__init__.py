"""
Isaac Gym Environment Classes

Import all environment implementations for easy access.
"""

from .base_environment import IsaacGymEnvironment
from .cartpole_environment import CartpoleEnvironment

# Environment registry for dynamic instantiation
ENVIRONMENT_REGISTRY = {
    "cartpole": CartpoleEnvironment,
    # Future environments can be added here
    # "ant": AntEnvironment,  
    # "humanoid": HumanoidEnvironment,
}

def create_environment(env_name: str, gym, sim, sim_params, num_envs: int, device: str, logger, isaac_gym_envs_path: str):
    """
    Factory function to create environment instances
    
    Args:
        env_name: Name of environment ("cartpole", "ant", "humanoid", etc.)
        gym: Isaac Gym instance
        sim: Isaac Gym simulation handle
        sim_params: Simulation parameters
        num_envs: Number of parallel environments
        device: Device for tensor operations
        logger: Logger instance
        isaac_gym_envs_path: Path to IsaacGymEnvs assets
        
    Returns:
        Environment instance
    """
    env_name_lower = env_name.lower()
    
    if env_name_lower not in ENVIRONMENT_REGISTRY:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(ENVIRONMENT_REGISTRY.keys())}")
    
    env_class = ENVIRONMENT_REGISTRY[env_name_lower]
    return env_class(gym, sim, sim_params, num_envs, device, logger, isaac_gym_envs_path)