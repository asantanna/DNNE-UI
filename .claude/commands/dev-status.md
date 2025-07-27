# Development Status

## Current Status: ✅ WORKING

PPO training with IsaacGymEnvs is fully functional. Training runs smoothly without freezes or hangs.

## Important Documentation

Before proceeding, familiarize yourself with these key documents:
- `docs-dnne/for_claude/` - Claude-specific documentation including performance guides
- `docs-dnne/architecture/adaptive-yielding.md` - Detailed explanation of the adaptive yielding system

## Recent Work Completed

### Node-Specific Command-Line Switches (✅ Implemented)
- Added disambiguation system for command-line arguments
- Syntax: `--epochs 10` (single node) or `--epochs 55:10 56:5` (node-specific)
- Fails fast with clear error when switches are ambiguous
- Node IDs now visible in UI titles (e.g., "Get Batch (50)")

### Checkpoint System Redesign (✅ Implemented)
- New cleaner argument structure:
  - `--save-checkpoint` - Enable checkpoint saving (flag only)
  - `--out-dir <dir>` - Specify output directory (default: "runs/singles")
  - `--load-checkpoint <dir>` - Load checkpoints from directory
- Respects both command-line flag AND node's `checkpoint_enabled` setting
- Fixed spurious checkpoint warnings

### Quality of Life Improvements (✅ Implemented)
- Suppressed deprecation warnings and FBX library warnings
- Timeout accepts plain numbers as seconds (e.g., `--timeout 5` = 5 seconds)
- Fixed device resolution for "auto" setting in Network nodes

## Environment Setup

```bash
# Activate conda environment
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# Export workflows using programmatic export script
cd /mnt/e/ALS-Projects/DNNE/DNNE-UI
python claude_scripts/programmatic_export.py [workflow_name]
# Examples:
python claude_scripts/programmatic_export.py                  # Defaults to Cartpole_PPO
python claude_scripts/programmatic_export.py Cartpole_PPO     # Export Cartpole_PPO
python claude_scripts/programmatic_export.py "MNIST Test"     # Export MNIST Test (note the space!)
python claude_scripts/programmatic_export.py Yield_Test       # Export Yield_Test

# Run exported workflow
cd export_system/exports/[workflow_name]
python runner.py --save-checkpoint --out-dir my_checkpoints --max-iterations 100 --headless
```

### Workflow Export

**Workflow Locations**:
- Saved workflows: `user/default/workflows/` (e.g., `Cartpole_PPO.json`, `MNIST Test.json`, `Yield_Test.json`)
- Exported code: `export_system/exports/{workflow_name}/` (spaces converted to underscores)

**Export Script**: `claude_scripts/programmatic_export.py`
- Accepts workflow name as parameter (use exact name including spaces)
- If no parameter provided, defaults to `Cartpole_PPO`
- Important: Use workflow names with spaces as saved (e.g., `"MNIST Test"` not `"MNIST_Test"`)
- The exporter automatically converts spaces to underscores in the output directory name

## Current Focus: Yield_Test Verification

### Overview
The Yield_Test workflow is a single workflow file containing two causally independent subgraphs:
- **MNIST subgraph**: The complete MNIST Test supervised learning network
- **PPO subgraph**: The complete Cartpole_PPO reinforcement learning network

These two subgraphs have no connections between them - they are completely independent computational paths within the same workflow. This design allows us to test whether DNNE's async queue framework properly interleaves execution between independent subgraphs, ensuring neither monopolizes compute resources.

### Understanding the Concurrency Model
- **MNIST nodes**: Use async/await with queue operations, naturally yielding when waiting for data
- **PPO Agent**: Runs synchronous rl_games code with explicit `sync_adaptive_yield()` calls
- The async queue mechanism provides natural concurrency for MNIST nodes

### Verification Plan

#### Step 1: Add Execution Tracking
1. Add logging to track when each workflow is actively computing
2. Log workflow switches to show interleaving pattern
3. Track queue wait times vs compute times

#### Step 2: Run Concurrent Test
1. Execute the Yield_Test workflow: `python runner.py --timeout 60s`
2. Observe that both workflows make progress:
   - MNIST epochs should advance
   - PPO episodes should complete
3. Neither workflow should starve the other

#### Step 3: Analyze Execution Pattern
1. Add timestamps to show execution interleaving
2. Verify that:
   - MNIST nodes yield naturally when waiting for data
   - PPO yields periodically during training loops
   - Both workflows get fair execution time

### Expected Behavior
With proper concurrent execution:
- MNIST nodes process batches between PPO environment steps
- PPO training yields periodically, allowing MNIST to progress
- Both workflows complete their tasks without blocking each other
- The async queue mechanism naturally provides concurrency for MNIST

### Key Files to Examine/Modify
1. `framework/globals.py` - Add execution tracking
2. `runner.py` - Add progress reporting for both workflows
3. Monitor PPO's `sync_adaptive_yield()` calls in rl_games_dnne
4. Track MNIST's natural async yielding via queue operations

### Phase 1: Simple PPO vs Non-PPO Time Tracking

**Key Insight**: Within the single Yield_Test workflow:
- The PPO subgraph is the only part that calls `sync_adaptive_yield()`
- The MNIST subgraph uses natural async/await yielding
- By measuring time in `sync_adaptive_yield()`, we can determine the execution balance

**Measurement Approach**:
- Time between `sync_adaptive_yield()` calls = PPO subgraph execution time
- Time during yield operations = MNIST subgraph execution time
- Total yields count = frequency of execution switching

**Expected Results**:
- Both subgraphs get significant execution time (neither at 0% or 100%)
- Frequent yielding demonstrates proper interleaving
- Stable percentages over time indicate fair resource sharing

### Future Measurement Ideas

Additional monitoring approaches for deeper analysis:

1. **Statistical Yield Analysis**
   - Distribution of yield intervals
   - Variance in execution chunks
   - Outlier detection for long-running sections

2. **Visual Progress Indicators**
   - Real-time execution balance display
   - Moving averages over time windows
   - Subgraph switching visualization

3. **Queue Activity Monitoring**
   - Track which nodes from each subgraph are active
   - Measure queue depths for both subgraphs
   - Identify bottlenecks within each subgraph

4. **Execution Pattern Analysis**
   - Record sequence of subgraph switches
   - Analyze switching patterns
   - Detect anomalous behavior

5. **Performance Impact Assessment**
   - Compare individual subgraph performance vs combined
   - Measure concurrency overhead
   - Optimize yield frequency

### Notes on Current Implementation
- Adaptive yielding currently acts as `sleep(0)` - sufficient for concurrency verification
- The single workflow design with independent subgraphs is ideal for testing execution fairness
- This approach validates DNNE's ability to handle complex workflows with parallel computation paths

## Lessons Learned

### Design Principles
- **Fail Fast**: Use NotImplementedError in base classes rather than guessed defaults
- **Clear Errors**: Provide specific error messages for ambiguous command-line switches
- **Respect User Settings**: Both node settings AND command-line flags must agree for features to activate

### Checkpoint Best Practices
- `saved_runs/` directory exists for checkpoints that should be preserved in git
- `export_system/exports/` is git-ignored and can be safely overwritten
- Always delete export directory before re-exporting to prevent stale files

### Command-Line Design
- Use `nargs='?'` with `const` for optional arguments with defaults
- Plain numbers should be accepted for common units (seconds for timeout)
- Node-specific syntax with colons provides clear disambiguation

## Key Files

### Export System
- `/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/graph_exporter.py` - Main export logic with argument parsing
- `/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/templates/framework/globals.py` - Global configuration with adaptive yield

### Frontend
- `/mnt/e/ALS-Projects/DNNE/DNNE-UI-Frontend/src/services/litegraphService.ts` - Node ID in titles
- `/mnt/e/ALS-Projects/DNNE/DNNE-UI-Frontend/src/scripts/app.ts` - Workflow loading with ID updates

## Debug Features

### Command Line Flags
- `--save-checkpoint`: Enable checkpoint saving
- `--out-dir <dir>`: Set output directory (default: runs/singles)
- `--load-checkpoint <dir>`: Load checkpoints from directory
- `--epochs N` or `--epochs 55:10 56:5`: Override epoch counts
- `--max-iterations N`: Override PPO iterations
- `--timeout 30` or `--timeout 5m`: Set run duration
- `--visual`/`--headless`: Control rendering
- `--inference`: Skip training

### Adaptive Yielding
- **MNIST nodes**: Natural yielding via async/await queue operations
- **PPO Agent**: Explicit yielding via `sync_adaptive_yield()` in rl_games_dnne
- **Framework**: Global class manages yielding with metrics tracking
- **Current Status**: PPO yielding confirmed working (tested that disabling it causes PPO to dominate)