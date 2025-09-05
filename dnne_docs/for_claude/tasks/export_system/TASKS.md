# Export System Tasks

*For historical accomplishments, see HISTORY.md*

## Current Status
🔧 **Active** - Multi-model SGDOptimizer support implemented, investigating gradient conflicts
- Fixed sync checking for multi-model optimizers
- Label node links removed after resolution for clean exports

## 📋 Active TODOs
- [ ] Resolve gradient conflict between multiple SGDOptimizers
  - SGDOptimizer 40 (3 control networks) conflicts with SGDOptimizer 81 (shadow network)
  - Error: InPlace operation on [128, 20] tensor in AsStridedBackward0
  - Works when SGDOptimizer 81 disabled

## Low Priority
- [ ] Make other IsaacGym environments DNNE-compatible (add dnne: sections to YAMLs)
- [ ] Export profiling and metrics (track time/size, add progress reporting)
- [ ] Custom node template support (user templates, validation framework)

## 💡 Quick Reference

### Test Commands
```bash
# Activate environment
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# Test all exports
python claude_scripts/test_all_exports.py

# Run unit tests
./dnne_test quick
```

### Export Structure
```
exports/{workflow_name}/
├── runner.py              # Main entry point
├── metadata.json          # Workflow metadata
├── framework/             # Queue framework
├── nodes/                 # Generated node code
└── custom_compute_funcs/  # Custom functions
```