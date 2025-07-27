# PPO Debug Status

## Current Status: ✅ WORKING

PPO training with IsaacGymEnvs is fully functional. Training runs smoothly without freezes or hangs.

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

# Test PPO export and run
cd /mnt/e/ALS-Projects/DNNE/DNNE-UI
python claude_scripts/programmatic_export.py Cartpole_PPO

# Run with new checkpoint system
cd export_system/exports/Cartpole_PPO
python runner.py --save-checkpoint --out-dir my_checkpoints --max-iterations 100 --headless
```

## Pending Tasks

### Current Todo List

| ID | Task | Priority | Status |
|----|------|----------|--------|
| 15 | Fix widget order mapping in PPO exporter comments | medium | pending |
| 35 | Research the adaptive part of adaptive_yield and implement | high | pending |
| 36 | Try a different IGE environment to verify it also works | medium | pending |
| 37 | Think about how to use IGE environment with non-PPO workflow | medium | pending |
| 39 | Make graph_exporter delete existing export directory before re-exporting to prevent old files from hanging around | high | pending |

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
- Currently disabled (returns immediately) - needs investigation
- Framework exists for queue-based adaptive delays
- Node starvation metrics tracked but not yet used for adaptation