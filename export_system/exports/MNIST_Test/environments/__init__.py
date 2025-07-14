"""Environment factory and base classes"""

from .base_environment import IsaacGymEnvironment
from .cartpole_environment import CartpoleEnvironment

# Environment registry
ENVIRONMENT_REGISTRY = {
    "Cartpole": CartpoleEnvironment,
    "cartpole": CartpoleEnvironment,
}

def create_environment(env_name: str, gym, sim, sim_params, num_envs: int, device: str, logger, isaac_gym_envs_path: str):
    """
    Factory function to create environment instances
    
    Args:
        env_name: Name of the environment to create
        gym: Isaac Gym instance
        sim: Isaac Gym simulation handle
        sim_params: Simulation parameters
        num_envs: Number of parallel environments
        device: Device for tensor operations
        logger: Logger instance
        isaac_gym_envs_path: Path to IsaacGymEnvs repository
        
    Returns:
        Environment instance
    """
    if env_name not in ENVIRONMENT_REGISTRY:
        raise ValueError(f"Unknown environment: {env_name}. Available environments: {list(ENVIRONMENT_REGISTRY.keys())}")
    
    env_class = ENVIRONMENT_REGISTRY[env_name]
    return env_class(gym, sim, sim_params, num_envs, device, logger, isaac_gym_envs_path)
