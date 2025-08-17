# DNNE Development Status

## !!IMPORTANT!! READ THESE IMPORTANT GUIDELINES !!
- **Coding Guidelines**: `.claude/commands/rules-for-DNNE.md`
- **Task Tracking**: `dnne_docs/for_claude/tasks/INDEX.md`

$ARGUMENTS$

*For historical development sessions, see HISTORY.md*

## Latest Achievements (2025-08-17)

### Tensor Constant Node ✅
- Implemented TensorNode for generating constant tensors in ML workflows
- Supports 9 initialization modes (zeros, ones, uniform, normal, Kaiming, Xavier, custom)
- Flexible dimension parsing ("10", "2,3", "[2,3,4]")
- Configurable dtype and seed for reproducibility
- Complete with export template and unit tests
- Successfully integrated in Franka_Coop_Nodes workflow

### Export System Schema Resolution ✅
- Fixed ConcatExporter missing get_initial_output_schema() method
- Fixed SplitExporter missing schema methods with proper size calculation
- Refactored to eliminate DRY violations in parse_split_positions()
- Fixed FAIL-FAST rule violations (no silent fallbacks)
- Franka_Coop_Nodes workflow export issues resolved

### Schema Format Enhancement ✅
- Support for simplified single-element schema format in YAML
- Split node exporter handles both `[x,x]` array and `x` number formats
- UI displays single elements aesthetically as `[x]` instead of `[x-x]`
- Full backward compatibility maintained

## Quick Reference

### Essential Commands
```bash
# Activate environment
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# Restart DNNE Server (runs on Windows)
use MCP function: "util_restart_dnne"

# Start Agent Client (WSL2)
python dnne_agent/dnne_agent_client.py

# Export workflow
python claude_scripts/programmatic_export.py MNIST_Test

# Run exported workflow
cd export_system/exports/MNIST_Test
python runner.py --epochs 10

# Test telemetry
./dnne_test telemetry

# Build frontend
./build_frontend.sh
```

### Key Ports
- DNNE UI: 8188
- Agent Server: 8767
- Agent Health: 8769
- Telemetry: 8770

### Key Documentation
- **Task Index**: `dnne_docs/for_claude/tasks/INDEX.md`
- **Architecture**: `dnne_docs/architecture/`
- **CLAUDE.md**: Project overview

## Claude Code Capabilities
- **Server Control**: Restart via `/remote_command` endpoint
- **Browser Automation**: UI interaction via MCP
- **WSL2 Access**: Server at `http://172.22.160.1:8188`

## Recent Commits
- Implement Tensor constant node with 9 initialization modes
- Add comprehensive unit tests for Tensor node
- Support simplified schema format (single number for single elements)
- Update Split node exporter for both schema formats
- Enhance IsaacGymEnvs display aesthetics for single elements
- Implement @dnne_node decorator system
- Auto-discovery for node registration and exporters

---
*Focus on active tasks in INDEX.md*