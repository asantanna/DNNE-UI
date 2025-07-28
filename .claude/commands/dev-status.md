# Development Status

## Current Status: ✅ WORKING

PPO training with IsaacGymEnvs is fully functional. Training runs smoothly without freezes or hangs. Concurrent execution of independent subgraphs (MNIST + PPO) verified working with thread-safe yielding.

## Important Documentation

Before proceeding, familiarize yourself with these key documents:
- `docs-dnne/for_claude/` - Claude-specific documentation including performance guides
- `docs-dnne/architecture/adaptive-yielding.md` - Detailed explanation of the adaptive yielding system
- `docs-dnne/experiments/yield_tests/` - Research on thread-safe yielding solution

## Recent Work Completed

### Thread-Safe Yielding Solution (✅ Implemented)
- **Problem**: `sync_adaptive_yield()` calling `loop._run_once()` from within a task caused RuntimeError
- **Solution**: Implemented thread-safe yielding using `run_in_executor` and `ThreadSafeYielder` singleton
- **Result**: PPO training runs in thread pool, yields control back to main event loop for MNIST execution
- **Files**: 
  - `export_system/templates/framework/globals_threadsafe.py` - Thread-safe yielding implementation
  - `export_system/templates/nodes/ppo_agent_queue.py` - Modified to use `run_in_executor`
- **Verification**: Yield_Test workflow shows concurrent MNIST and PPO execution

### Export System Improvements (✅ Implemented)
- Directory cleaning before export (prevents stale files)
- Concurrency report for workflows with PPO nodes
- All debug prints tagged with `#DBG_TAG#` for easy enable/disable
- MNIST epoch tracker cleaned up (no debug prefix, added "Training starting..." message)

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

## Debug Features

### Enabling/Disabling Debug Prints
All debug prints are tagged with `#DBG_TAG#` at the end of lines:

```bash
# Enable all debug prints:
find . -name "*.py" -exec sed -i 's/# \(.*\) #DBG_TAG#$/\1/' {} \;

# Disable all debug prints:
find . -name "*.py" -exec sed -i 's/^\(.*print.*\)$/# \1 #DBG_TAG#/' {} \;
```

### Command Line Flags
- `--save-checkpoint`: Enable checkpoint saving
- `--out-dir <dir>`: Set output directory (default: runs/singles)
- `--load-checkpoint <dir>`: Load checkpoints from directory
- `--epochs N` or `--epochs 55:10 56:5`: Override epoch counts
- `--max-iterations N`: Override PPO iterations
- `--timeout 30` or `--timeout 5m`: Set run duration
- `--visual`/`--headless`: Control rendering
- `--inference`: Skip training

## Concurrent Execution: Yield_Test Workflow

### Overview
The Yield_Test workflow contains two causally independent subgraphs:
- **MNIST subgraph**: Complete supervised learning network
- **PPO subgraph**: Complete reinforcement learning network

### Concurrency Model
- **MNIST nodes**: Natural async yielding via queue operations
- **PPO Agent**: Runs in thread pool with thread-safe yielding
- **ThreadSafeYielder**: Singleton manages yield requests from threads
- **Result**: Both subgraphs execute concurrently without blocking

### Verification Results
With thread-safe yielding:
- MNIST epochs progress normally
- PPO training runs with visual cartpole simulation
- Frequent "[SYNC_YIELD] In thread context" messages show yielding is active
- Concurrency report shows balanced execution between subgraphs

## Key Implementation Details

### Thread-Safe Yielding Architecture
1. **PPO runs in thread**: `await loop.run_in_executor(None, run_training_with_yielding)`
2. **Thread requests yield**: Puts request in queue to main loop
3. **Main loop processes**: Yields via `await asyncio.sleep(delay)`
4. **MNIST executes**: During PPO's yield periods
5. **Cycle repeats**: Ensures fair execution time for both subgraphs

### Adaptive Yielding
- Dynamic delay calculation based on node starvation metrics
- Ranges from 0ms (minimal) to 10ms (aggressive)
- Tracks execution balance between PPO and non-PPO nodes
- `Global.print_concurrency_report()` shows execution percentages

## Lessons Learned

### Asyncio Execution Model
- **Cannot call `loop._run_once()` from within a task** - causes RuntimeError
- **Solution**: Use `run_in_executor` to isolate synchronous code in threads
- **Thread-safe communication**: Use queues to communicate between threads and main loop

### Design Principles
- **Fail Fast**: Use NotImplementedError in base classes rather than guessed defaults
- **Clear Errors**: Provide specific error messages for ambiguous command-line switches
- **Clean Output**: Production code should have minimal debug output

### Export Best Practices
- Always clean export directory before re-exporting
- Tag debug code with `#DBG_TAG#` for easy management
- Keep concurrency reports and other useful diagnostics

## Key Files

### Export System
- `/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/graph_exporter.py` - Main export logic with directory cleaning
- `/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/templates/framework/globals.py` - Adaptive yielding implementation
- `/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/templates/framework/globals_threadsafe.py` - Thread-safe yielding
- `/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/templates/nodes/ppo_agent_queue.py` - PPO with thread execution

### Frontend
- `/mnt/e/ALS-Projects/DNNE/DNNE-UI-Frontend/src/services/litegraphService.ts` - Node ID in titles
- `/mnt/e/ALS-Projects/DNNE/DNNE-UI-Frontend/src/scripts/app.ts` - Workflow loading with ID updates

### External Dependencies
- `/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs/isaacgymenvs/train.py` - Modified for inline DNNE_print
- `/home/asantanna/DNNE-LINUX-SUPPORT/rl_games_dnne/` - Custom RL games with adaptive yielding

## Next Steps

### 1. Test Concurrency Metrics Printouts
- **Goal**: Verify the concurrency balance report shows meaningful data
- **Method**: Run Yield_Test workflow and check the concurrency report output
- **Expected**: Should show PPO vs MNIST execution time percentages
- **Location**: `Global.print_concurrency_report()` at end of execution

### 2. Wire Concurrency Metrics to Nodes
- **Goal**: Enable adaptive yielding based on actual node starvation metrics
- **Current**: Adaptive delay calculation exists but node metrics aren't being updated
- **Required Changes**:
  - Add `Global.update_node_execution(node_id)` calls in node compute methods
  - Track queue depths with `Global.update_queue_pressure(total_queued)`
  - Ensure `_compute_adaptive_delay()` uses real metrics instead of returning 0
- **Result**: Dynamic yielding that responds to actual execution patterns

### 3. Validate Adaptive Algorithm
- **Test Cases**:
  - Heavy PPO load → should increase yield delays
  - Heavy MNIST load → should decrease PPO yield delays
  - Balanced load → should stabilize at minimal delays
- **Metrics to Monitor**:
  - Max starvation time per node
  - Queue depths
  - Yield frequency and duration