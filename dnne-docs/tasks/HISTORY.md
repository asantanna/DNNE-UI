# DNNE Task History

## Completed Features (Jan 2025)

### Session: Type System Refactoring & New Nodes
**Date**: Jan 14, 2025

#### Completed:
1. **Type System Refactoring**
   - Changed LOSS_TENSOR to LOSS_SCALAR throughout codebase
   - Updated CrossEntropyLoss, TrainingStep, EpochTracker nodes
   - Updated frontend color configurations in dark.json and dnneColors.ts
   - Added SCALAR type with training color

2. **Concat Utility Node**
   - 4 optional tensor inputs (input_a through input_d)
   - Two modes: "wait for all" and "as available"
   - Pad modes: "pad with zeros" and "hold previous"
   - State management for hold previous mode
   - Full export system support

3. **GeometricLoss Node**
   - 5 error metrics: Max Abs Error, Euclidean Dist, Manhattan Dist, KL Div, Norm KL Div
   - Normalized KL Divergence (0-1 scale) using log(n) normalization
   - Created math_utils.py with reusable metric functions
   - Implemented proper dependency system for framework files

4. **Export System Enhancement**
   - Modified _copy_dependency to support any relative path from templates/
   - Framework dependencies now properly placed in framework directory
   - Dependency system tested with math_utils.py

5. **UI Fixes**
   - Disabled auto-rewiring on node deletion
   - Fixed input connector outlines (removed HollowCircle shape)
   - Fixed server URL display (shows localhost instead of 0.0.0.0)
   - Removed unused Queue and Model Library sidebar tabs

### Previous Sessions
See git history for earlier completed work.