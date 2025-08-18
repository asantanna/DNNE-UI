# Franka Cooperative Control Workflow Tasks

## Current Status
✅ **Franka_Coop_Nodes exports and runs** - Robot moves continuously without crashes
✅ **TENSOR STANDARDS ENFORCED** - All nodes now follow strict dimension conventions (2025-08-18)
✅ **GRADIENT FLOW FIXED** - Isaac Gym observations properly support gradient computation
✅ **DEVICE HANDLING UNIFIED** - All nodes use global device configuration consistently
🎉 **READY FOR REAL RESEARCH** - Major cleanup complete, fail-fast philosophy implemented

## 📚 Documentation
See: [`dnne_docs/experiments/franka_coop_nodes/franka_coop_overview.md`](../../experiments/franka_coop_nodes/franka_coop_overview.md)

---

## Tensor Dimension Standards (NEW - 2025-08-18)
Per CLAUDE.md update:
- **Dimension 0**: ALWAYS batch/environment dimension (N samples or num_envs)
- **Dimension 1**: ALWAYS feature dimension (F features)  
- **Dimension 2+**: Additional data dimensions (H/W for images, L for sequences)
- **NEVER** use 1D tensors except for scalar losses/rewards
- **FAIL-FAST**: Nodes must raise `ValueError` if dimensions don't match convention
- **Concat/Split**: ONLY operate on dim=1 (features), never dim=0 (batch)

---

## ✅ COMPLETED - Comprehensive Node Fixes (2025-08-18)

### Core Tensor Operations (Critical Priority)
1. ✅ Fix concat_exporter.py - enforce dim=1, tensor dimensions, device handling, gradient preservation
2. ✅ Fix concat_node_queue.tpl - enforce dim=1, tensor dimensions, device handling, gradient preservation
3. ✅ Fix split_exporter.py - enforce dim=1, tensor dimensions, device handling, gradient preservation
4. ✅ Fix split_node_queue.tpl - enforce dim=1, tensor dimensions, device handling, gradient preservation

### Network & Layer Nodes (High Priority)
5. ✅ Fix network_exporter.py - enforce tensor dimensions, device handling, gradient preservation
6. ✅ Fix network_queue.tpl - enforce tensor dimensions, device handling, gradient preservation
7. ✅ Fix linear_layer_exporter.py - enforce tensor dimensions, device handling, gradient preservation (virtual node)
8. ✅ Fix linear_layer_queue.tpl - enforce tensor dimensions, device handling, gradient preservation (N/A)

### Training Components (High Priority)
9. ✅ Fix training_step_exporter.py - enforce tensor dimensions, device handling, requires gradients
10. ✅ Fix training_step_queue.tpl - simplified to fail-fast, removed unnecessary checks

### Dataset & Data Nodes
11. ✅ Fix get_batch_queue.tpl - added device placement and gradient enabling
12. ✅ Fix tensor_queue.tpl - added device placement and gradient enabling
13. ✅ Fix data_streamer_queue.tpl - batch format [1, features], device, gradients
14. ✅ Fix custom_computation_queue.tpl - global device config, removed type checking

### Isaac Gym Integration
15. ✅ Fix isaac_gym_sim_queue.tpl - added detach().requires_grad_(True) for clean gradient boundary
    - Observations treated as data source like DataStreamer
    - Creates leaf tensors for gradient computation

### Loss & Optimizer Nodes (Already Good)
16. ✅ cross_entropy_queue.tpl - already handles scalar loss correctly
17. ✅ sgd_optimizer_queue.tpl - already correct, no tensor output

### Control Flow Nodes (No Changes Needed)
18. ✅ or_node_queue.tpl - async already reasonable
19. ✅ balancer/epoch_tracker - control flow only, no tensors

## Remaining Minor Tasks

### AsyncIO & System Issues
1. ⬜ Investigate why await asyncio.sleep(0) is needed in certain places
2. ⬜ Improve async efficiency in nodes that override run() - OR and Concat

### Testing Infrastructure
3. ⬜ Implement robust timeout mechanism in runner.py using thread with CauseExitException
   - Current timeout mechanism is unreliable with async code
   - Implement using separate thread that sleeps then raises CauseExitException
   - Ensures clean shutdown even if main loop is stuck

### Documentation
46. ✅ Update TASKS files with current todo list (DONE)

---

## Fix Criteria for Each Node

### Standard Tensor Nodes Must:
1. **Enforce dimension standards**: `[batch, features, ...]` format
2. **Device consistency**: Output on same device as input
3. **Gradient preservation**: Maintain gradients in training mode
4. **FAIL-FAST**: Raise ValueError on dimension mismatches
5. **No 1D tensors**: Except for scalars (losses/rewards)

### Example Validation Code:
```python
# At start of compute()
if input_tensor.dim() < 2:
    raise ValueError(f"Expected tensor with at least 2 dimensions [batch, features], got shape {input_tensor.shape}")

# Ensure device consistency
output = output.to(input_tensor.device)

# In training mode, gradients should flow (unless in inference mode which wraps in torch.no_grad())
```

---

## Original Tasks (Keep for Reference)

### Joint Locking Implementation  
**Status**: 🔴 Not Started  
**Priority**: MEDIUM (after core fixes)  
**TODO**: Implement PD control for joints 3-6 (currently free-floating with zero torque)

### Training Experiments
**Status**: 🔴 Not Started  
**Priority**: HIGH (after gradient issues resolved)  
**TODO**: Run training and monitor coordination emergence between the 3 controllers

---

## Quick Reference

**Workflow Structure**:
- 31 nodes, 45 connections
- 3 parallel networks (128→128→1)
- Split 56: `target_pos, eef_pos`
- Split 45: `joint_theta[0], [1], [2]`
- Concat 42: mode="as available"

**Key Files**:
- Workflow: `user/default/workflows/Franka_Coop_Nodes.json`
- Loss: `user/default/custom_compute_funcs/franka_coop_nodes_loss.py`
- Analyzer: `claude_scripts/analyze_workflow.py`