# Resume PPO Debug Session

You are resuming work on debugging the DNNE PPO implementation. In DNNE, the workflow is "Cartpole_PPO". In IsaacGymEnvs, the environment is "cartpole". Read these critical context files to understand the current state:

## Required Reading

```
@docs-dnne/code-quality-checklist.md
@docs-dnne/for_claude/debug_strategy_for_ppo.md  
@docs-dnne/for_claude/dnne_debug_improvements.md
```

## Useful Tools

1. **DNNE vs IsaacGymEnvs Performance Profiler**: @claude_scripts/profiling/performance_profiler.py [debug flags]
2. **DNNE workflow exporter**: @claude_scripts/programmatic_export.py
3. **NOTE**: Due to a bug in CLAUDE CODE, files sometimes disappear mysteriously. You can check that it happened by doing git status and verifying it now appears as deleted. You can restore from git but often this loses your current work. One thing you can do is make one change at a time to that file and backup the file to /tmp so that you don't lose all your work again.

## Current Status

We are debugging why DNNE's PPO implementation doesn't learn properly compared to IsaacGymEnvs. The strategy is to make both implementations deterministic and compare their execution step-by-step to find divergences. NOTE: we have broken something with regards to running IsaacGymEnvs under the key tool we use in 

## Key Findings So Far

1. **Initial Environment Reset**: We think that DNNE does not reset the environment initially like IsaacGymEnvs. We tried to add code to do it is now commented out because it was causing all sorts of CUDA and tensor errors.
2. **Debug Infrastructure**: Added `[DNNE_DEBUG]` markers for filtering for our debug messages more easily. We should be using the function DNNE_print() to print debug messages to guarantee this prefix is always present.
3. **Remaining Issues**: 
   - Network weight initialization differences
   - Observation normalization (RunningMeanStd) initialization
   - Action divergence from different network/normalization initialization

## Next Steps

1. Find out why running IsaacGymEnvs "train.py" script works when we run it directly but doesn't learn correctly under the profiler. We added some debug code in there which may have broken things?
2. Focus on matching the network initialization and observation normalization between DNNE and IsaacGymEnvs to eliminate the action divergence that starts from the first step.
