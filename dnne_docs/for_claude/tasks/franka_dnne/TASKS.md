# FrankaDNNE Environment Integration

## Status
✅ COMPLETED - FrankaDNNE environment now supports manual reset control and target-based episode completion

## Completed (2025-08-20)
- ✅ Disabled auto-reset in FrankaDNNE `post_physics_step`
- ✅ Added manual `reset()` method for trigger-based resets
- ✅ Verified IsaacGymSim node sends "done" trigger when episode ends
- ✅ Verified IsaacGymSim node handles "reset" input trigger correctly
- ✅ Added distance-based done trigger (10cm threshold from target)
- ✅ Added debug output when target is reached

## Key Changes
1. **franka_dnne.py**: 
   - Disabled auto-reset in `post_physics_step` (lines 638-642)
   - Added manual `reset()` method
   - Added target distance check in `compute_reward()`
   - Episodes end on timeout OR reaching target

2. **IsaacGymSim Node**:
   - Properly emits "done" trigger when `done.any()` is True
   - Handles both auto-reset (`reset_when_done=True`) and manual reset
   - Accepts "reset" input trigger for workflow-controlled resets

## Usage
- Set `reset_when_done=False` in IsaacGymSim node for manual control
- Wire "done" output to decision logic
- Wire "reset" input from control logic
- Target reached threshold: 10cm from gripper to target

---

## Debug Visualization Feature (2025-09-08)

### Status
✅ **Complete** - Feature implemented and test script created (2025-09-08)

### Purpose
Add debug sphere visualization for predicted end-effector positions to help debug Franka_Coop_V2's shadow network predictions.

### Key Requirements
- Debug sphere must be **visual only** (no physics interactions)
- Activated only when `extra_args` dictionary contains debug data
- Default behavior unchanged when `extra_args=None`
- Gray sphere similar to target sphere but smaller (radius ~0.03)

### Task List

#### Phase 1: Environment Setup
- [x] **Create debug sphere actor in FrankaDNNE.__init__**
  - Created gray sphere asset with `disable_gravity=True` and `fix_base_link=True`
  - Set collision_filter=0 to prevent ALL physics interactions
  - Initially positioned sphere out of view at z=-10
  - Stored debug sphere actor handle as `self._debug_sphere_id`

#### Phase 2: Step Method Updates  
- [x] **Modify FrankaDNNE.pre_physics_step() signature**
  - Added `extra_args=None` parameter
  - Check for `"debug_sphere_pos"` key in extra_args dict
  - Update debug sphere position using `set_actor_root_state_tensor_indexed`
  - Visual-only with no physics interactions

- [x] **Update VecTask.step() signature**
  - Added `extra_args=None` parameter to base class
  - Pass through to pre_physics_step with inspection check
  - Maintains backward compatibility

#### Phase 3: Export System Integration
- [x] **Update IsaacGymSim node template**
  - Modified compute() to check for action.extra_args attribute
  - Pass extra_args to env.step()
  - Added usage documentation for workflows

#### Phase 4: Testing & Validation
- [x] **Test visual-only behavior**
  - Created test script `claude_scripts/test_debug_sphere.py`
  - Tests sphere movement in circle pattern
  - Verifies no physics interactions
  - Tests hiding when extra_args=None

- [ ] **Integration test with Franka_Coop_V2**
  - Pass shadow network predictions as debug positions
  - Verify visual feedback accuracy
  - Ensure no performance impact when disabled

### Implementation Notes

#### Collision Prevention
```python
# Set collision filter to interact with nothing
debug_sphere_opts.collision_group = 0  # No collision group
debug_sphere_opts.collision_mask = 0   # Collide with nothing
```

#### Actor State Updates
```python
# Update only position, preserve orientation/velocity
self._debug_sphere_state[0, 0:3] = torch.tensor(pos, device=self.device)
```

#### Extra Args Format
```python
extra_args = {
    "debug_sphere_pos": [x, y, z],  # Required for sphere visualization
    # Future debug features can add more keys
}
```

### Success Criteria
1. Debug sphere visible at specified coordinates
2. No physics interactions with any actors
3. Zero overhead when `extra_args=None`
4. Clean integration with export system
5. Helpful for debugging Franka_Coop_V2 predictions