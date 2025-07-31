# DNNE Development Status

## Current Task: Fix Command-Line Interface Inconsistencies
**Status**: 🔧 Multiple UI/UX issues identified

### Immediate Tasks:
1. **Fix Yield_Test_Async workflow** - Update CrossEntropyLoss nodes to match new 2-input structure
2. **Fix command-line inconsistencies** - Standardize argument formats across all switches
3. **Fix balancing node titles** - Add node_id to title display

### Current Todo List:
- [x] Export all workflows using the export_all_workflows script
- [ ] Balancing node not showing node_id in title
- [ ] Fix inconsistent command-line argument formats
- [ ] Standardize on comma separators for all multi-value arguments
- [ ] Make --verbose/--debug recognize node IDs directly without node. prefix
- [ ] Update help text to show consistent comma separators and node ID format

## Recent Accomplishments (2025-01-30)
**Status**: ✅ Major features completed!

### 🔧 Node Interface Fixes
- ✅ Moved weight initialization from Network to LinearLayer nodes
- ✅ Added per-layer weight initialization with auto-detection (Kaiming for ReLU, Xavier for tanh/sigmoid)
- ✅ Fixed CrossEntropyLoss to original 2-input design (predictions, labels)
- ✅ Changed SGDOptimizer input from 'network' to 'model' for clarity
- ✅ Restored EpochTracker to original 3-input design (epoch_stats, loss, accuracy)
- ✅ Fixed MetricsLogger to use WARNING level (suppress INFO messages)
- ✅ Successfully exported and tested MNIST_Test and CIFAR10_Test workflows

### 📁 Version Control Improvements
- ✅ Updated .gitignore to properly track default workflows
- ✅ Added all 7 workflow JSON files to version control
- ✅ Committed all node interface changes

### 🎨 Node Color System
- ✅ Created centralized `node_colors.py` with color constants for all node types
- ✅ Updated server.py to send COLOR and BGCOLOR in node_info API
- ✅ Frontend now applies node colors from backend automatically
- ✅ All nodes have consistent, visually distinct colors by category

### 📊 CIFAR-10 Dataset Support
- ✅ Implemented CIFAR10Dataset node for loading CIFAR-10 image datasets
- ✅ Created exporter and queue template for CIFAR-10
- ✅ Added schema resolution methods to ExportableNode base class
- ✅ Fixed Yield_Test_Async workflow with CIFAR-10 integration

### 🔢 Universal Node ID Display
- ✅ All nodes now show their ID in titles (e.g., "Linear Layer (42)")
- ✅ Solved the mystery of where IDs were being added (change tracker + auto-save)
- ✅ Implemented consistent ID display for new nodes and loaded workflows

### 🛠️ Export System Improvements
- ✅ Updated all node exporters to use new schema resolution approach
- ✅ Created `export_all_workflows.py` script for batch exports
- ✅ Fixed various export issues and removed deprecated methods

### 🔄 Yield API Updates
- ✅ Updated PPO training to use new unified yield API
- ✅ Added subgraph="ppo" and is_item_ref parameters to all yield calls

## Current Issues

### 🐛 Known Bugs
1. **Yield_Test_Async workflow** - Still has old CrossEntropyLoss structure with 3 inputs (needs fixing to 2)
2. **Command-line inconsistencies**:
   - Some switches use comma separators (`--verbose mnist,queue`)
   - Others use space separators (`--epochs 55:10 56:20`)
   - Verbose requires "node." prefix (`--verbose node.10`) while epochs doesn't (`--epochs 55:10`)
3. **Balancing node titles** - Don't show node ID in title (should show "Balancing Node (10)")

## Previous Work: Execution Balance in Concurrent Subgraphs
**Status**: 📋 Implementation plan documented

We documented the plan for proper subgraph-based metrics. Previous work:
- ✅ Implemented subsystem-specific logging (`--debug balancing`)
- ✅ Fixed periodic balance reports (every 10s)
- 🔍 Discovered current metrics measure PPO CPU time vs yield time (not actual MNIST execution)
- 📋 Documented implementation plan in system-balancing.md

**Next Steps** (see `docs-dnne/architecture/system-balancing.md#implementation-plan---metrics-collection`):
1. Implement subgraph-based metrics collection
2. Track throughput (items/sec) for all nodes
3. Track CPU time for sync nodes only
4. Report total async time to identify starvation

## Quick Reference

### Essential Commands
```bash
# Activate environment
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# Export single workflow
cd /mnt/e/ALS-Projects/DNNE/DNNE-UI
python claude_scripts/programmatic_export.py Yield_Test

# Export ALL workflows
python claude_scripts/export_all_workflows.py

# Run with balance debugging (shows reports every 10s)
cd export_system/exports/Yield_Test
python runner.py --debug balancing --timeout 60s

# Toggle debug prints on/off
python claude_scripts/toggle_DBG_TAG.py <filename>
```

### Command-Line Switches for `runner.py`
- `--debug <subsystems>` - Enable debug for specific subsystems (e.g., `balancing,yield`)
- `--verbose <subsystems>` - Enable info-level logging for subsystems
- `--timeout 30s` or `--timeout 5m` - Set run duration
- `--save-checkpoint --out-dir <dir>` - Enable checkpoint saving
- `--epochs 55:10 56:20` - Node-specific settings (nodeID:value)
- `--no-yield` - Disable adaptive yielding (for testing)

### Available Logging Subsystems
- `balancing` - Execution balance reports between subgraphs
- `yield` - Adaptive yielding timing and decisions
- `ppo` - PPO training details
- `mnist` - MNIST training progress
- `queue` - Queue operations and flow
- `checkpoint` - Checkpoint save/load operations

### Important Locations
- Workflows: `/mnt/e/ALS-Projects/DNNE/DNNE-UI/user/default/workflows/`
- Exports: `/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/`
- Templates: `/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/templates/`
- RL Games: `/home/asantanna/DNNE-LINUX-SUPPORT/rl_games_dnne/`

### Key Documentation
- **System Balancing**: `docs-dnne/architecture/system-balancing.md` - Philosophy and implementation plan for subgraph-based metrics
- **Adaptive Yielding**: `docs-dnne/architecture/adaptive-yielding.md` - Current yielding implementation (needs update)
- **Logging Guidelines**: `docs-dnne/development/logging-guidelines.md` - Subsystem logging standards

## Todo List

### Short-term (Current Session)
- [x] **Understand execution metrics** - 86/14 split is PPO CPU vs yield time (not MNIST time)
- [x] **Document implementation plan** - Added to system-balancing.md
- [ ] **Implement subgraph-based metrics** - Track actual throughput per subgraph

### Medium-term (This Week)
- [ ] **Add real-time progress display** - Show training progress every 5-10s
- [ ] **Test --no-yield flag** - Verify PPO dominates without yielding
- [ ] **Add subgraph activity tracking** - Track which subgraph is active with timestamps
- [ ] **Test verbose output** - Monitor detailed execution patterns
- [ ] **Update adaptive-yielding.md** - Currently marked as "hopelessly wrong"

### Long-term (Future Improvements)
- [ ] **Fix Hydra disable_existing_loggers** - Find proper solution instead of workaround
- [ ] **Create visual progress indicators** - Nice-to-have execution visualization
- [ ] **Fix total yields counter** - Currently shows 0 despite yields happening
- [ ] **Wire metrics to adaptive algorithm** - Make yielding respond to actual starvation

## Feature Requests
*Detailed specs in docs-dnne/future/*

### 🧠 ML Features
- [ ] **ConvNet Support** (Medium/Medium) - Add MaxPool2D, BatchNorm2D, Flatten nodes for CNN architectures
- [ ] **Advanced Optimizers** (Medium/Medium) - Add Adam, AdamW, RMSprop optimizers and LR schedulers
- [ ] **Data Pipeline** (Medium/Medium) - Custom datasets, data augmentation, validation splitting

### 🤖 Robotics Features  
- [ ] **PPO Decomposition** (Low/Large) - Split PPOAgent into Actor, Critic, Trainer, Buffer nodes
- [ ] **Rate Limiting** (Low/Small) - Add Hz control to GetBatch for fixed-frequency robotics

### 🔧 System Features
- [ ] **UI Feedback** (Low/Medium) - Progress bars, connection status, better error messages
- [ ] **Linux Test Suite** (Low/Large) - Automated testing framework for Linux compatibility
- [ ] **Architecture Refactor** (Low/Ongoing) - Technical debt and queue framework improvements

### 📊 Visualization Features
- [ ] **Training Dashboards** (Low/Medium) - Real-time loss graphs, metrics visualization

---

### Key Implementation Notes

**Thread-Safe Yielding**: PPO runs in thread pool with `run_in_executor`, uses ThreadSafeYielder singleton to request yields from main loop. MNIST runs naturally async.

**Logging Workaround**: Hydra disables loggers after configuration. We re-enable in a2c_common.py before use. Proper fix: configure Hydra with `disable_existing_loggers: False`.

**Balance Measurement**: Currently tracks PPO CPU time vs yield time via `sync_adaptive_yield()` calls. Will be updated to track per-subgraph metrics (throughput and CPU time where applicable). See implementation plan in system-balancing.md.