# rl_games Surgical Integration Plan for DNNE

## Objective
Replace DNNE's custom PPO implementation with rl_games' proven PPO components while maintaining DNNE's async queue coordination and multi-network capabilities.

## Problem Statement
- **Need**: Identical PPO behavior to IsaacGymEnvs for learning comparison
- **Constraint**: Cannot hand over control to rl_games monolithic training loop
- **Requirement**: Must maintain DNNE's cooperative async scheduling for multi-network workflows

## Solution: Surgical Component Extraction

### What We Extract from rl_games
1. **PPO Algorithm Components**:
   - `calc_gradients()` - PPO loss and gradient computation
   - `discount_values()` - GAE advantage computation  
   - `train_actor_critic()` - Wrapper around calc_gradients
   - Loss functions from `common_losses.py`

2. **Data Structures & Utilities**:
   - PPODataset for minibatch creation
   - Parameter naming conventions
   - Tensor format requirements

### What We Keep from DNNE
1. **Async Queue Coordination**: Incremental data collection via queues
2. **Cooperative Scheduling**: Yield control after each PPO update
3. **Multi-network Support**: Other training nodes continue processing
4. **Visual Workflow**: Export system and node-based architecture

## Implementation Plan

### Phase 1: Documentation and Research ✅
- [x] Save surgical integration plan to docs
- [x] Research rl_games PPO component extraction details
- [x] Map parameter names and method signatures

### Phase 2: Component Extraction
- [ ] Extract `calc_gradients()` method from `a2c_continuous.py`
- [ ] Extract `discount_values()` method from `a2c_common.py`
- [ ] Extract loss functions from `common_losses.py`
- [ ] Create standalone PPO component wrapper

### Phase 3: Integration
- [ ] Create parameter mapping from DNNE to rl_games terminology
- [ ] Build adapter layer for DNNE to rl_games data format
- [ ] Replace PPOTrainerNode custom implementation
- [ ] Maintain async queue coordination pattern

### Phase 4: Validation
- [ ] Test surgical integration maintains async queue coordination
- [ ] Validate identical behavior to IsaacGymEnvs Cartpole
- [ ] Performance testing with multi-network workflows

## Key Findings from Research

### Parameter Mapping
| DNNE Parameter | rl_games Equivalent | Default |
|----------------|-------------------|---------|
| `gae_lambda` | `tau` | 0.95 |
| `clip_param` | `e_clip` | 0.2 |
| `value_coef` | `critic_coef` | 4.0 |
| `ppo_epochs` | `mini_epochs_num` | 8 |
| `entropy_coef` | `entropy_coef` | 0.0 |

### Required Input Format for rl_games Methods
```python
input_dict = {
    'old_values': torch.Tensor,      # Previous value estimates
    'old_logp_actions': torch.Tensor, # Previous log probabilities
    'advantages': torch.Tensor,       # GAE advantages
    'returns': torch.Tensor,          # Discounted returns
    'actions': torch.Tensor,          # Actions taken
    'obs': torch.Tensor,              # Observations
    'mu': torch.Tensor,              # Previous action means
    'sigma': torch.Tensor,           # Previous action stds
}
```

### Core Methods to Extract
1. **calc_gradients(input_dict)** from `a2c_continuous.py`
2. **discount_values(...)** from `a2c_common.py`
3. **actor_loss()** and **critic_loss()** from `common_losses.py`

## Expected Outcome
- ✅ **Identical PPO Algorithm**: Same as IsaacGymEnvs/rl_games
- ✅ **Maintained Concurrency**: Multiple networks train simultaneously  
- ✅ **Cooperative Scheduling**: Fair compute time distribution
- ✅ **Minimal Architecture Changes**: Keep DNNE's async queue system

## Current Pattern (Keep)
```python
# Async queue coordination - UNCHANGED
async def compute(self, state, policy_output, reward, done, model):
    self.buffer_states.append(state)  # Incremental collection
    
    if len(self.buffer_states) >= self.horizon_length:
        loss = self.ppo_update(...)  # <- REPLACE WITH RL_GAMES
        self.reset_buffer()
        return {"loss": loss, "training_complete": signal}
```

## Target Pattern (After Integration)
```python
# Async queue coordination - UNCHANGED
async def compute(self, state, policy_output, reward, done, model):
    self.buffer_states.append(state)  # Incremental collection
    
    if len(self.buffer_states) >= self.horizon_length:
        loss = self.rlgames_ppo_update(...)  # <- RL_GAMES COMPONENTS
        self.reset_buffer()
        return {"loss": loss, "training_complete": signal}
```

This surgical approach provides rl_games' proven PPO implementation while preserving DNNE's multi-network capabilities and async architecture.