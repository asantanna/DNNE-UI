# DNNE Debugging Guide: Making DNNE Match IsaacGymEnvs

## Executive Summary

This guide captures the lessons learned from debugging DNNE's PPO implementation to match IsaacGymEnvs (IGE) behavior exactly. The key achievement was fixing DNNE's exported Cartpole_PPO code to initialize and execute identically to IGE's proven implementation.

**Core Principle**: "Don't guess - instrument and compare." When DNNE and IGE diverge, add matching debug prints to both systems until you spot the exact point of divergence.

## Debugging Philosophy & Approach

### The Matching Debug Prints Pattern

The most effective debugging technique was adding identical debug prints to both DNNE and IGE:

```python
# In both DNNE and IGE code:
print(f"[DNNE_DEBUG] PPO_CYCLE: Step {step}: action={action:.4f}, value={value:.4f}, reward={reward:.4f}")
```

This immediately reveals:
- Execution order differences
- Value divergences
- Missing initialization steps
- Incorrect tensor shapes

### Systematic Comparison Methodology

1. **Enable debug mode in both systems**: `PPO_CYCLE_DEBUG=1`
2. **Run with fixed seed**: Ensures deterministic execution
3. **Compare outputs line by line**: Look for first divergence
4. **Instrument the divergence point**: Add more detailed prints
5. **Fix and repeat**: Until outputs match exactly

### Inheritance Over Reimplementation

**Key Insight**: DNNE should inherit from IGE implementations with minimal overrides rather than reimplementing from scratch.

Example:
```python
class CartpoleDNNE(Cartpole):  # Inherit from IGE's Cartpole
    # Only override what's necessary for DNNE's async architecture
```

## Architecture Understanding

### DNNE vs IGE Execution Models

**IGE**: Synchronous training loop
```
Initialize → Reset → Loop { Step → Collect → Train }
```

**DNNE**: Async queue-based coordination
```
Nodes communicate via queues → Each node yields control → Cooperative scheduling
```

### The Export System

DNNE's export system converts visual workflows to standalone Python scripts:
```
Visual Workflow (JSON) → Graph Exporter → Templates → Generated Python Code
```

This affects debugging because:
- Errors may be in templates, not just generated code
- Need to fix both template and regenerate
- Export process can introduce subtle bugs

### Key Components

1. **PPOAgentNode**: Generates actions from observations
2. **PPOTrainerNode**: Collects trajectories and performs PPO updates
3. **IsaacGymEnvNode**: Manages environment initialization
4. **IsaacGymStepNode**: Steps the physics simulation
5. **CartpoleActionNode**: Converts PPO actions to environment format

## Critical Discoveries & Fixes

### 1. Bootstrap Value for GAE

**Problem**: Index out of bounds when computing advantages

**Root Cause**: GAE (Generalized Advantage Estimation) needs the value of the state AFTER the last action in the trajectory

**Fix**: Collect bootstrap value before PPO update
```python
# Get value of current state (after last action)
with torch.no_grad():
    last_features = model['shared'](last_state)
    last_values = model['value'](last_features).squeeze(-1)

# Pass to GAE computation with shape [horizon_length + 1, num_envs]
values_with_bootstrap = torch.cat([values, last_values.unsqueeze(0)], dim=0)
```

### 2. Tensor Reshaping Pattern

**Problem**: Incorrect tensor shapes for minibatch creation

**Root Cause**: rl_games expects specific reshape pattern called "swap_and_flatten01"

**Fix**: Transform [horizon, envs, features] → [envs, horizon, features] → [envs*horizon, features]
```python
def swap_and_flatten(tensor):
    if tensor.dim() == 3:
        return tensor.transpose(0, 1).reshape(batch_size, -1)
```

### 3. Action Format Differences

**Problem**: CartpoleActionNode returned dictionary with forces instead of raw tensor

**Root Cause**: Misunderstanding of Isaac Gym's action interface

**Fix**: Return raw action tensor directly
```python
# Wrong
return {"action": {"forces": scaled_actions}}

# Correct
return {"action": action_tensor}
```

### 4. Trigger Mode Handling

**Problem**: IsaacGymStepNode stuck in trigger-only mode

**Root Cause**: PPOTrainerNode sends "collecting" signals that were treated as triggers

**Fix**: Differentiate signal types
```python
if trigger is not None and trigger.get('signal_type') != 'collecting':
    # Use trigger mode
else:
    # Use normal mode
```

### 5. Initial Observation Reset

**Problem**: DNNE started with all-zero observations while IGE had proper initial state

**Root Cause**: IGE calls reset() during initialization, DNNE didn't

**Fix**: Add initial reset after environment creation
```python
# In CartpoleDNNE initialization
self.reset()  # Critical for proper initial observations
```

## Debug Tools & Techniques

### Environment Variables

- `PPO_CYCLE_DEBUG=1`: Enables detailed PPO cycle logging
- `USE_RL_GAMES_DNNE=1`: Makes IGE use instrumented rl_games version
- `FIXED_SEED=42`: Forces deterministic execution

### Debug Print Categories

1. **PPO_CYCLE**: Training loop progress
2. **PPO_BATCH**: Batch preparation for training
3. **PPO_GRAD**: Gradient computation details

### Instrumentation Strategy

```python
# Add to both DNNE and IGE at same locations:
if os.environ.get('PPO_CYCLE_DEBUG', '0') == '1':
    print(f"[DNNE_DEBUG] PPO_CYCLE: {description}")
    print(f"[DNNE_DEBUG] Tensor shape: {tensor.shape}")
    print(f"[DNNE_DEBUG] Values: min={tensor.min():.4f}, max={tensor.max():.4f}")
```

### Using rl_games_dnne

Created an instrumented fork of rl_games that includes debug prints:
```bash
# Make IGE use instrumented version
USE_RL_GAMES_DNNE=1 python isaacgymenvs/train.py task=Cartpole
```

## Integration Patterns

### Environment Handle Pattern

Pass complete environment context between nodes:
```python
env_handle = {
    "environment": self.env,  # CartpoleDNNE instance
    "gym": self.gym,
    "sim": self.sim,
    "viewer": self.viewer
}
```

### Queue-Based Async Coordination

Maintain DNNE's async architecture while using IGE components:
```python
async def compute(self, state, policy_output, reward, done, model):
    # Collect data incrementally
    self.buffer_states.append(state)
    
    if len(self.buffer_states) >= self.horizon_length:
        # Use rl_games components for PPO update
        loss = self.rlgames_ppo_update(...)
        return {"loss": loss, "training_complete": signal}
```

### rl_games Component Extraction

Instead of reimplementing PPO, extract proven components:
```python
from rl_games_dnne.dnne_exports import PPOComponents, RunningMeanStd

# Use rl_games GAE computation
advantages = self.ppo_components.discount_values(rewards, values, dones)

# Use rl_games loss computation
train_result, loss = self.ppo_components.train_actor_critic(input_dict, model)
```

## Lessons Learned

### 1. Always Match Initialization Sequences

- IGE resets environment after creation, DNNE must too
- Network initialization order matters
- Observation normalization must match

### 2. Debug Prints Must Be Identical

- Same format, same location, same values
- Use consistent prefixes like [DNNE_DEBUG]
- Include both systems in output for easy comparison

### 3. Don't Reimplement Proven Algorithms

- Use IGE's VecTask instead of custom environment management
- Extract rl_games PPO instead of writing from scratch
- Inherit and override minimally

### 4. Instrument Rather Than Guess

- When something doesn't work, add debug prints
- Compare with working implementation
- Let data guide fixes, not assumptions

### 5. Test Incrementally

- Fix one issue at a time
- Verify each fix before moving on
- Use epoch limits to test quickly

### 6. Understand Tensor Lifecycle

- Know when tensors need reshaping
- Understand dimension conventions (batch first vs time first)
- Track device placement carefully

## Common Pitfalls to Avoid

1. **Assuming tensor shapes**: Always print and verify
2. **Skipping initialization steps**: Compare full startup sequence
3. **Ignoring bootstrap values**: GAE needs future state values
4. **Mixing sync and async patterns**: Keep clear separation
5. **Debugging without reference**: Always compare to working implementation

## Quick Debugging Checklist

When DNNE doesn't match IGE:

- [ ] Enable PPO_CYCLE_DEBUG in both systems
- [ ] Run with fixed seed for determinism
- [ ] Compare initialization sequences
- [ ] Check tensor shapes at each step
- [ ] Verify action format matches environment expectations
- [ ] Ensure bootstrap value collection for GAE
- [ ] Look for missing reset() calls
- [ ] Compare buffer sizes and reshape patterns
- [ ] Check if trigger mode is stuck
- [ ] Verify device placement consistency

## Conclusion

Successfully debugging DNNE to match IGE required systematic comparison, careful instrumentation, and understanding both architectures deeply. The key was never guessing but always comparing with the working reference implementation. When in doubt, add more debug prints until the divergence becomes obvious.

The resulting system maintains DNNE's innovative async queue architecture while leveraging IGE's proven implementations, achieving the best of both worlds.