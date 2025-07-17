# Isaac Gym Reorganization Plan

## Abbreviations
- **IG**: Isaac Gym (NVIDIA's physics simulation library)
- **IGE**: IsaacGymEnvs (NVIDIA's RL environments built on Isaac Gym)

## Overview
Reorganize DNNE's Isaac Gym integration to use IsaacGymEnvs' proven environment implementations while maintaining DNNE's flexible node-based architecture.

## Motivation
- Current DNNE implementation has CUDA initialization errors
- Reimplementing VecTask initialization logic is complex and error-prone
- Better to adapt IGE's battle-tested code for DNNE's async architecture

## New Directory Structure

```
custom_nodes/
├── utility_nodes/
│   ├── __init__.py
│   └── or_node.py                    # Generic OR node (moved from isaac_gym_nodes)
│
└── robotics_nodes/
    ├── isaac_gym_base_nodes.py       # IsaacGymEnvNode, IsaacGymStepNode
    ├── gym_envs/
    │   ├── __init__.py
    │   ├── base_env_dnne.py         # Base class for DNNE environments
    │   ├── cartpole_dnne.py         # CartpoleDNNE adapted from IGE
    │   ├── ant_dnne.py              # Future: AntDNNE
    │   └── humanoid_dnne.py         # Future: HumanoidDNNE
    └── action_nodes/
        ├── __init__.py
        ├── cartpole_action_node.py
        ├── ant_action_node.py
        └── continuous_action_base.py # Base for continuous control
```

## Implementation Plan

### Phase 1: Move Generic Components
1. Create `utility_nodes` directory
2. Move ORNode from isaac_gym_nodes.py to utility_nodes/or_node.py
3. Update node registration

### Phase 2: Create New Isaac Gym Structure
1. Create `gym_envs` and `action_nodes` directories
2. Create `isaac_gym_base_nodes.py` with generic env/step nodes
3. Move CartpoleActionNode to action_nodes/

### Phase 3: Implement CartpoleDNNE
1. Copy IsaacGymEnvs' cartpole.py to gym_envs/cartpole_dnne.py
2. Inherit from IGE's Cartpole class
3. Add DNNE-specific adaptations:
   ```python
   class CartpoleDNNE(Cartpole):
       def step_async(self, actions):
           """DNNE-compatible async step"""
           self.pre_physics_step(actions)
           self.gym.simulate(self.sim)
           self.gym.fetch_results(self.sim, True)
           self.post_physics_step()
           return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
       
       def set_custom_reward_fn(self, reward_fn):
           """Allow custom reward computation"""
           self.custom_reward_fn = reward_fn
   ```

### Phase 4: Update Base Nodes

#### IsaacGymEnvNode Updates:
- Import environments from gym_envs/
- Output environment handle:
  ```python
  env_handle = {
      "environment": self.env_instance,  # CartpoleDNNE instance
      "gym": self.gym,
      "sim": self.sim,
      "viewer": self.viewer
  }
  ```

#### IsaacGymStepNode Updates:
- Accept environment handle input
- Use env.step_async() instead of manual stepping
- No more manual tensor management

### Phase 5: Update Export System
1. Update node_exporters paths
2. Update template imports
3. Register new node locations

### Phase 6: Test & Verify
1. Test OR node from new location
2. Verify CartpoleDNNE initialization (should fix CUDA errors)
3. Test PPO training with new structure

### Phase 7: Cleanup
1. Delete isaac_gym_nodes.py
2. Remove old environment implementations
3. Update all imports

## Key Design Patterns

### Environment Handle Pattern
```python
# IsaacGymEnvNode outputs full environment instance
outputs = {
    "env_handle": {
        "environment": cartpole_env,  # CartpoleDNNE instance
        "gym": self.gym,
        "sim": self.sim,
        "viewer": self.viewer
    },
    "observations": initial_obs
}
```

### Custom Reward Pattern
```python
class CustomRewardNode(QueueNode):
    async def compute(self, env_handle, observations, original_rewards):
        # Compute custom rewards
        custom_rewards = my_reward_function(observations)
        return {"rewards": custom_rewards}
```

### Workflow Flexibility
1. **Standard**: IsaacGymEnvNode → IsaacGymStepNode → PPO nodes
2. **Custom Rewards**: Insert reward override node
3. **Custom Actions**: Environment-specific action conversion
4. **Multi-Environment**: Different environments to shared PPO

## Benefits
1. **Immediate Fix**: Solves CUDA initialization errors
2. **Proven Code**: Uses IGE's VecTask implementation
3. **Modular**: Easy to add new environments
4. **Flexible**: Custom rewards/observations via nodes
5. **Maintainable**: Clear separation of concerns

## Migration Notes
- All current DNNE IGE code is committed, safe to reorganize
- Start with minimal CartpoleDNNE implementation
- Add features incrementally
- Test each phase before proceeding