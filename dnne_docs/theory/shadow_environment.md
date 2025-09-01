# Shadow Environment: Differentiable Control Through Non-Differentiable Simulators

## The Problem

In RL workflows with physical simulators, we face two critical problems:

### Problem 1: Temporal Alignment (Solved)
- Networks process obs(t) to produce action(t)
- Loss needs obs(t+1) which arrives later
- **Solution**: Eat_N and Barrier nodes create proper synchronization

### Problem 2: The Gradient Canyon
- SGD needs to backpropagate from loss to network parameters
- Path: Loss ← obs(t+1) ← **Simulator** ← action(t) ← Networks
- **Simulator is non-differentiable** - gradients cannot flow through
- Networks can never learn because they never receive gradients

## The Shadow Environment Solution

Create a differentiable neural network that learns to approximate the simulator's dynamics.

### Architecture

Two parallel paths:
1. **Real Path**: obs(t) → Networks → action → Real Simulator → real_obs(t+1)
2. **Shadow Path**: [obs(t), action] → Shadow Env Network → pred_obs(t+1)

Two losses:
1. **Control Loss**: Computed from real_obs(t+1) value
2. **Prediction Loss**: MSE(pred_obs(t+1), real_obs(t+1))

### Key Insight: Gradient Bridge

- Use real_obs(t+1) to compute loss VALUE
- Use pred_obs(t+1) for gradient PATH
- Shadow env provides differentiable bridge across the simulator canyon

## Bootstrap Process

### Phase 1: Random Actions
- Control networks output random actions (untrained)
- Shadow env learns dynamics from these random trajectories
- Supervised learning works even with random data

### Phase 2: Noisy Signal
- Shadow env achieves some accuracy
- Provides noisy but directional gradients to control networks
- Think: pred_obs ≈ real_obs + α*noise where α is large

### Phase 3: Signal Emergence
- Control networks start learning from noisy gradients
- SNR improves as shadow env gets better
- α decreases over time

### Phase 4: Specialization
- Control networks improve → action distribution changes
- Shadow env specializes to this new distribution
- Positive feedback loop: better shadow → better control → better shadow

## Gradient Quality vs Accuracy

**Critical insight**: Shadow env doesn't need perfect accuracy, just similar gradients.

If real dynamics: f(obs, action) → obs'
And shadow dynamics: f̂(obs, action) → obs' + ε

As long as ∂f/∂action ≈ ∂f̂/∂action in direction, control can learn.

## Implementation Notes

### Requirements
- Maximum observability (joint positions, velocities, EE pose, targets)
- Smooth dynamics assumption (true for robot control)
- Careful gradient flow management (stop_gradient on real_obs)

### Training Strategy
- Simultaneous training of shadow and control networks
- Shadow env always has clear supervised signal
- Control networks initially learn nothing, then slowly improve

### Potential Issues
- **Distribution shift**: Random → controlled actions
- **Gradient quality**: Good prediction ≠ good gradients
- **Compounding errors**: Multi-step predictions drift
- **Local minima**: Early bad gradients might trap networks

### Optimization Ideas
- Gradient regularization for shadow env
- Action noise injection for exploration
- Residual learning (predict Δobs instead of obs')
- Pre-training shadow env if needed
- Replay buffer with diverse experiences
- Curriculum learning (start with easier tasks)
- GAN discriminator to match dynamics "style"
- Sensitivity regularization (ensure ∂obs/∂action accuracy)

## Integration with Temporal Alignment

Complete workflow combines both solutions:

1. Eat_N/Barrier handle temporal synchronization
2. Shadow env provides gradient bridge
3. Together enable end-to-end differentiable control

```
obs(t) → Barriers → Networks → action
           ↓                      ↓
      (hold until)         [obs(t), action]
           ↓                      ↓
    SGD.step_complete      Shadow Env
                                 ↓
                           pred_obs(t+1)
                                 ↓
                         (gradient path back)
```

## Why This Should Work

1. Supervised learning (shadow env) is reliable and monotonic
2. Robot dynamics are smooth and differentiable
3. Gradient noise is tolerable (SGD is robust)
4. Simple task (Franka gripper) should be learnable
5. Modular design allows debugging each component

## Open Questions

- Optimal shadow env architecture size?
- Need for recurrence/memory?
- Best loss combination weights?
- How to detect/handle gradient pathologies?
- When to switch from random to learned actions?

## Implementation Issues & Solutions

### Critical Issues Discovered (2025-08-31)

#### 🔴 Issue 1: Shadow Loss Computing Wrong Values
**Problem**: `franka_coop_nodes_shadow_loss.py` is comparing target positions between actual and predicted observations. Target position is fixed and not affected by the simulator - it's an input, not a dynamic state!

**Current (WRONG)**:
```python
target_pos = obs[..., 0:3]           # This never changes!
pred_target_pos = pred_obs[..., 0:3] # Network can't affect this!
distance = torch.norm(pred_target_pos - target_pos, p=2, dim=-1)
```

**Solution**: Compare dynamic elements that actually change:
- `eef_pos` (indices 3-5): End-effector position - PRIMARY
- `joint_theta` (indices 10-18): Joint angles - SECONDARY

**Proposed Loss**:
```python
# Extract dynamic elements
actual_eef = obs[..., 3:6]
pred_eef = pred_obs[..., 3:6]
actual_joints = obs[..., 10:19]  
pred_joints = pred_obs[..., 10:19]

# Compute separate losses
eef_loss = torch.norm(pred_eef - actual_eef, p=2, dim=-1)
joint_loss = torch.norm(pred_joints - actual_joints, p=2, dim=-1)

# Normalize to similar scales (eef in meters, joints in radians)
# Typical ranges: eef [-1, 1]m, joints [-3.14, 3.14]rad
joint_loss = joint_loss / 3.14  # Normalize to ~[-1, 1] range

# Weighted combination (eef_pos is more important)
total_loss = 0.8 * eef_loss + 0.2 * joint_loss
```

#### 🔴 Issue 2: Shadow Network Should Only Predict Dynamic Elements
**Problem**: Shadow network predicts full observation including static elements (target_pos, eef_quat). This wastes capacity and creates confusion.

**Static Elements** (should pass through unchanged):
- `target_pos` (0-2): Fixed target location
- `eef_quat` (6-9): Can be derived from joint angles
- `episode_time` (19): Deterministic counter

**Dynamic Elements** (should be predicted):
- `eef_pos` (3-5): Changes based on actions
- `joint_theta` (10-18): Direct result of torque commands

**Solution**: Hybrid observation construction
1. Shadow network only predicts: `[eef_pos, joint_theta]` (12 values)
2. Use Split/Concat nodes to build hybrid observation:
   - Extract static from actual: `[target_pos, eef_quat, episode_time]`
   - Extract dynamic from predicted: `[eef_pos, joint_theta]`
   - Concat into full observation for Main Loss

**Workflow Changes Needed**:
```
Isaac Gym → Split → [static_elements, dynamic_elements]
                            ↓
Shadow Network → predicts [pred_eef_pos, pred_joint_theta]
                            ↓
Concat([static_elements, pred_dynamics]) → Main Loss
```

#### 🔴 Issue 3: Control SGDs Must Not Modify Shadow Network
**Problem**: Shadow network has its own SGD optimizer, but control networks also backpropagate through it via Main Loss. This creates conflicting gradients!

**Critical**: Shadow network should ONLY be trained by its prediction loss, not by control loss. However, gradients must still flow THROUGH the shadow network to reach control networks.

**Solution: ContextVar-Based Gradient Isolation**

Using Python's `contextvars` module to track which optimizer initiated the backward pass:

```python
from contextvars import ContextVar

# Global context variable to track current optimizer
CURRENT_OPT_ID = ContextVar("CURRENT_OPT_ID", default=None)

# Context manager for SGD optimizer
def optimizer_ctx(opt_id):
    class Ctx:
        def __enter__(self): 
            self.token = CURRENT_OPT_ID.set(opt_id)
        def __exit__(self, *exc): 
            CURRENT_OPT_ID.reset(self.token)
    return Ctx()

# Hook registration in Network node
def zero_grad_if_unauthorized(module: nn.Module, authorized_id: str):
    """Register hooks that zero gradients from unauthorized optimizers"""
    handles = []
    for p in module.parameters():
        def make_hook():
            def hook(grad):
                # Check if current optimizer is authorized
                current_opt = CURRENT_OPT_ID.get()
                if current_opt == authorized_id:
                    return grad  # Allow gradient update
                else:
                    return torch.zeros_like(grad)  # Block update but pass through
            return hook
        handles.append(p.register_hook(make_hook()))
    return handles
```

**How it works**:
1. Each Network node registers gradient hooks with its authorized SGD optimizer ID
2. SGD optimizer sets context variable before calling `loss.backward()`
3. During backward pass, hooks check if current optimizer matches authorized ID
4. Unauthorized optimizers get zero gradients (no weight update) but gradient flow continues
5. Authorized optimizer gets normal gradients and updates weights

**Implementation in DNNE**:
```python
# In Network node initialization:
handles = zero_grad_if_unauthorized(self.module, f"SGD({self.sgd_node_id})")

# In SGD optimizer step:
with optimizer_ctx(f"SGD({self.node_id})"):
    loss.backward()  # Context identifies this SGD as the active optimizer
self.optimizer.step()
```

**Advantages**:
- **Selective Updates**: Only authorized optimizer can modify each network
- **Gradient Flow Preserved**: Control networks still receive gradients through shadow network
- **Async-Safe**: ContextVars handle task isolation automatically
- **Clean Design**: No global state, each network-optimizer pair is independent

### TODO List

#### High Priority (Blocking Issues)
- [x] Fix shadow loss to compare eef_pos and joint_theta (DONE: franka_coop_nodes_shadow_loss_fixed.py)
- [ ] Implement ContextVar gradient isolation in Network and SGDOptimizer nodes
- [ ] Modify shadow network to only predict dynamic elements

#### Medium Priority (Improvements)
- [ ] Determine optimal loss weight ratio (eef vs joints)
- [ ] Add normalization for different value ranges
- [ ] Consider adding velocity predictions for smoother dynamics

#### Low Priority (Future Enhancements)
- [ ] Add recurrence/memory to shadow network
- [ ] Implement curriculum learning (simple → complex motions)
- [ ] Add uncertainty estimation to predictions

### Decisions Made

1. **Shadow Loss Focus**: Will primarily optimize for end-effector position (80%) with joint angles as secondary (20%)

2. **Static vs Dynamic Split**: Shadow network will only predict physically dynamic elements, static elements pass through unchanged

3. **Gradient Isolation**: Will use ContextVar-based gradient hooks to allow gradient flow while preventing unauthorized weight updates

### Dynamic Range Analysis

**Observation Element Ranges**:
- `target_pos`: [-0.5, 0.5] meters (workspace bounds)
- `eef_pos`: [-1.0, 1.0] meters (reachable space)
- `eef_quat`: [-1, 1] (normalized quaternion)
- `joint_theta`: [-3.14, 3.14] radians (joint limits)
- `episode_time`: [0, max_episode_length] seconds

**Normalization Strategy**:
- EEF position: Already in reasonable range, use as-is
- Joint angles: Divide by π to get [-1, 1] range
- This ensures losses are comparable in magnitude