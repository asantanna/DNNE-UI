"""
Isaac Gym Simulator node simulator for deadlock analysis.
IsaacGym nodes can bootstrap with null actions to start the simulation loop.
"""

from typing import Dict, Any, Optional
from .base_node_sim import BaseNodeSimulator, NodeState

class IsaacGymSimulator(BaseNodeSimulator):
    """
    Simulates Isaac Gym simulation node behavior.
    
    Isaac Gym nodes:
    - Can bootstrap with null action to start simulation
    - Receive actions from the policy network
    - Step the simulation
    - Produce observations and done signals
    """
    
    def __init__(self, node_id: str, node_config: Dict[str, Any]):
        super().__init__(node_id, node_config)
        
        # Isaac Gym configuration
        self.can_bootstrap = True  # Can start with null action
        self.bootstrapped = False
        self.num_envs = node_config.get('num_envs', 1)
        
        # Input/output configuration
        self.inputs_required = {'action'}  # Action from policy
        self.outputs = {'observation', 'done', 'reward', 'info'}
        
        # Simulation state
        self.step_count = 0
        self.episode_count = 0
        
    def should_bootstrap(self) -> bool:
        """Check if this node should bootstrap with null action"""
        return self.can_bootstrap and not self.bootstrapped and 'action' not in self.inputs_available
        
    def bootstrap(self) -> Dict[str, Any]:
        """
        Bootstrap simulation with null action.
        This produces the first observation to break initial deadlock.
        """
        self.bootstrapped = True
        self.step_count = 1
        self.logger.info(f"Bootstrapping IsaacGym with null action")
        
        # Generate initial observation
        initial_obs = {
            'type': 'observation',
            'source_node': self.node_id,
            'step': 0,
            'num_envs': self.num_envs,
            'metadata': {
                'bootstrap': True,
                'null_action': True
            }
        }
        
        return {
            'observation': initial_obs,
            'done': {'all_done': False, 'env_dones': [False] * self.num_envs},
            'reward': {'rewards': [0.0] * self.num_envs},
            'info': {'step': 0, 'bootstrapped': True}
        }
        
    def can_execute(self) -> bool:
        """
        IsaacGym can execute when:
        1. It has an action input, OR
        2. It can bootstrap (first step with null action)
        """
        if 'action' in self.inputs_available:
            return True
        return self.should_bootstrap()
        
    def execute(self) -> Dict[str, Any]:
        """
        Step the simulation.
        
        Returns:
            Dict with observation, done, reward, and info
        """
        if not self.can_execute():
            raise RuntimeError(f"IsaacGym {self.node_id} cannot execute: no action and cannot bootstrap")
            
        self.state = NodeState.EXECUTING
        
        # Check if this is a bootstrap execution
        if self.should_bootstrap():
            return self.bootstrap()
            
        # Normal execution with action
        action = self.inputs_available.get('action')
        self.step_count += 1
        
        self.logger.info(f"Stepping simulation (step {self.step_count})")
        
        # Generate observation and other outputs
        observation = {
            'type': 'observation',
            'source_node': self.node_id,
            'step': self.step_count,
            'num_envs': self.num_envs,
            'metadata': {
                'action_received': True
            }
        }
        
        # Simulate done signals (some environments might finish)
        env_dones = [False] * self.num_envs
        # Randomly mark some as done for realistic simulation
        if self.step_count > 10 and self.step_count % 100 == 0:
            env_dones[0] = True  # First env done every 100 steps
            
        done = {
            'all_done': all(env_dones),
            'env_dones': env_dones,
            'step': self.step_count
        }
        
        # Generate rewards
        reward = {
            'rewards': [1.0] * self.num_envs,  # Dummy rewards
            'step': self.step_count
        }
        
        # Info dict
        info = {
            'step': self.step_count,
            'episode': self.episode_count,
            'action_processed': True
        }
        
        # Update episode count if any env is done
        if any(env_dones):
            self.episode_count += 1
            
        return {
            'observation': observation,
            'done': done,
            'reward': reward,
            'info': info
        }
        
    def reset(self):
        """Reset to initial state"""
        super().reset()
        self.bootstrapped = False
        self.step_count = 0
        self.episode_count = 0
        
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information for debugging"""
        info = super().get_state_info()
        info.update({
            'can_bootstrap': self.can_bootstrap,
            'bootstrapped': self.bootstrapped,
            'should_bootstrap': self.should_bootstrap(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'num_envs': self.num_envs
        })
        return info