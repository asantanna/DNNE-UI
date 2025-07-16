# DNNE Performance Work Log

## 2025-01-12 - Initial Analysis & Documentation Setup

### What We Did
- Created performance documentation structure in `docs-dnne/for_claude/`
- Analyzed current performance: 13.4 FPS vs 32,000 FPS baseline (2,388x slower)
- Identified main bottleneck: PPO Agent forward pass at 60ms

### Key Findings
- PPO Agent has redundant device transfers on already-cuda tensors
- Isaac Gym Step template missing smart throttling (export has it, template doesn't)
- Queue sizes default to 2, causing unnecessary blocking
- Profiling shows device transfer overhead in multiple nodes

### Conclusions
- Device transfer optimization could save 10-20ms
- Smart throttling already proven to work in exported code
- Queue framework needs optimization for high-throughput scenarios

### Next Steps
1. Fix device transfers in PPO Agent and Trainer templates
2. Update Isaac Gym Step template with smart throttling
3. Increase queue buffer sizes
4. Re-export and test performance

---

## 2025-01-15 - Fixed Seed Debugging Implementation

### What We Did
Implemented a fixed-seed debugging strategy to make both DNNE and IsaacGymEnvs produce deterministic results and identify where their implementations diverge.

### Implementation Details
1. **Added `--fixed-seed` argument** to graph_exporter.py and runner.py
2. **Added deterministic initialization code**:
   - Set seeds for torch, numpy, random
   - Enabled cudnn.deterministic mode
   - Disabled cudnn.benchmark
3. **Added debug logging** to PPO agent, PPO trainer, and Isaac Gym step nodes
4. **Created comparison scripts** to run both systems with same seed

### Key Discoveries
1. **Zero Observations Bug**: DNNE initially showed all zero observations
   - Root cause: Environment wasn't being reset after creation
   - Fix: Added environment reset call after initialization
   - Result: Observations now non-zero after step 5

2. **Repeating Reward Issue**: First 5 steps show constant reward (0.9987652897834778)
   - Hypothesis: Running mean statistics initialization issue
   - The reward calculation suggests small non-zero values in state
   - Could be caused by uninitialized or improperly updated running statistics
   - we should check how IsaacGymEnvs implements running mean statistics and make
     sure we are doing it identically

3. **Technical Issues Encountered**:
   - CUDA resource handle error when running comparisons
   - System instability requiring reboot
   - Timeout command issues with long-running processes

### Current State
- Fixed seed implementation is working
- Environment initialization partially fixed (non-zero observations after step 5)
- First 5 steps still show anomalous behavior
- Need to investigate running mean statistics initialization

### Files Modified
- `export_system/graph_exporter.py` - Added fixed seed argument
- `export_system/templates/nodes/ppo_agent_queue.py` - Added debug logging
- `export_system/templates/nodes/ppo_trainer_queue.py` - Added debug logging  
- `export_system/templates/nodes/isaac_gym_env_queue.py` - Added environment reset after creation
- `export_system/templates/nodes/isaac_gym_step_queue.py` - Added debug logging and temporary exit
- `claude_scripts/compare_ppo_implementations.py` - Comparison script

### Next Steps (After Reboot)
1. Fix the first 5 steps zero observation issue
2. Investigate running mean statistics initialization
3. Compare full execution traces between DNNE and IsaacGymEnvs
4. Identify exact divergence point in implementations

---

## 2025-01-15 - PPO Algorithm Alignment Work

### What We Did
1. **Fixed action parameter passing** - PPO Agent now outputs `action_mean` and `action_std` separately instead of concatenated `action_params`
2. **Updated templates** - Both exported code and templates now have the fix
3. **Verified rl_games integration** - PPO Trainer is using rl_games components correctly
4. **Ran performance test** - DNNE is 3x faster than IsaacGymEnvs (87k vs 28k steps/sec)

### Key Findings
1. **Performance is excellent** - DNNE runs at 87,186 steps/sec vs IsaacGymEnvs at 28,422 steps/sec
2. **Learning metrics missing** - DNNE is not tracking episode returns, making it impossible to verify if learning is working
3. **Algorithm components aligned** - We have:
   - rl_games PPO components integrated
   - Observation normalization fixed (previous commit)
   - Value function normalization added (previous commit)
   - Action parameters now passed correctly

### Current Issues
1. **Episode return tracking** - DNNE shows 0 episodes completed, 0 episode returns
2. **Fixed-seed comparison** - Comparison script runs but captures no debug output from either system
3. **CUDA errors** - Getting illegal memory access errors when running with GPU

### Next Steps
1. Debug why episode returns aren't being tracked in DNNE
2. Add episode return logging to base environment or Isaac Gym environment
3. Fix debug output capture in comparison script
4. Once episode tracking works, verify learning performance matches IsaacGymEnvs

---
EOF < /dev/null