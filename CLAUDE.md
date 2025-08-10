# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DNNE** (Distributed Neural Network Editor) is a visual programming environment for building neural networks and robotics control systems. It's based on ComfyUI's architecture but adapted for machine learning and robotics applications instead of image generation. The project transforms ComfyUI's visual node-based interface from a diffusion model platform into a comprehensive ML/robotics development environment with code export capabilities.

### Key Innovation
The primary innovation is the **export system** that converts visual node graphs into standalone, production-ready Python modules that can run on WSL2, GPU cloud providers (like Lambda), or integrate with robotics simulators like NVIDIA Isaac Gym. Unlike ComfyUI, DNNE doesn't execute graphs directly - instead, it exports them as Python scripts designed to run in tight loops with simulators.

### Repository Structure

#### Code Locations
- Backend code is checked out to: `/home/asantanna/DNNE/DNNE-UI`
- Front end code is checked out to: `/home/asantanna/DNNE/DNNE-UI-Frontend`
- Linux support code is checked out to: `/home/asantanna/DNNE/DNNE-LINUX-SUPPORT`

#### Backend Repository (This Repository)
Contains the main DNNE-UI backend with:
- `server.py` - Modified ComfyUI server that handles export functionality
- `custom_nodes/` - UI node implementations for ML and robotics
- `export_system/` - Export system that converts visual graphs to Python code
- `claude_scripts/` - Claude-created utility and test scripts for development

#### Frontend Repository
**GitHub**: https://github.com/asantanna/DNNE-UI-Frontend.git
Vue.js-based frontend providing the visual graph editor interface (replaces original ComfyUI frontend).

- To rebuild the frontend, use "./build_frontend.sh"

## Development Commands

- When using the Bash tool, do not redirect stderr because it prevents pipes from working.

### Environment Setup
The project requires a properly configured conda environment with PyTorch. To activate it:
```bash
source /home/asantanna/miniconda/bin/activate DNNE_PY38
```

**Note**: The standard `conda activate DNNE_PY38` command may not work in all shell contexts. Use the full path activation method above for reliable environment activation.

If the conda environment is not activated, you may encounter errors like:
- `ModuleNotFoundError: No module named 'torch'`
- Missing dependencies that are installed in the conda environment

### Isaac Gym Integration
IsaacGym and IsaacGymEnvs are installed and verified working:
- **IsaacGym**: `/home/asantanna/DNNE-LINUX-SUPPORT/isaacgym` - Core physics simulation library
- **IsaacGymEnvs**: `/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs` - Pre-built reinforcement learning environments
- **rl_games_dnne**: `/home/asantanna/DNNE-LINUX-SUPPORT/rl_games_dnne` - Replacement for rl_games with additional features and debug messages
- **Import Order**: Always import `isaacgym` before `torch` to avoid conflicts
- **GPU Support**: Verified working with CUDA and GPU PhysX acceleration
- **Environment Testing**: Cartpole and other environments tested successfully

### Starting the Server (Windows only!)
```bash
dnne.bat [args]
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Testing Export System
```bash
python /home/asantanna/DNNE/DNNE-UI/claude_scripts/programmatic_export.py
```

### Running Exported Scripts
After exporting a workflow, run the generated script:
```bash
cd export_system/exports/{workflow_name}
python runner.py
```
Note: Ensure the conda environment is activated before running scripts in `claude_scripts`.

### Common Development Tasks
- **Export workflows to Python**: Use the export system via the UI or programmatically through `claude_scripts/programmatic_export.py`
- **Add new node types**: Implement in `custom_nodes/` directory with `*_visnode.py` naming pattern

## Architecture Overview

### ⚠️ CRITICAL: WebSocket Communication Architecture ⚠️
**DNNE uses WebSocket for ALL client-server communication. NEVER create REST API endpoints for dynamic features.**
- All UI updates go through WebSocket
- All data requests use WebSocket messages (e.g., `request_logs`, not `/api/logs`)
- REST is ONLY for static resources and initial page load
- See `docs-dnne/architecture/websocket-not-rest.md` for details

### Core System Structure
- **Entry Point**: `main.py` - Initializes DNNE server with ML/robotics extensions
- **Node System**: `nodes.py` - Base DNNE node classes and robotics type integration
- **Server**: `server.py` - Web API and interface

### Custom Node Categories

All nodes are implemented in `custom_nodes/` directory with `*_visnode.py` naming pattern and organized by category:

#### ML Nodes (Category: "ml")
- **Data Nodes**: MNIST dataset, CIFAR-10 dataset, batch sampling, data loading
- **Layer Nodes**: Linear layers, Conv2D, activation functions, dropout, batch normalization, flatten
- **Training Nodes**: Cross-entropy loss, accuracy metrics, SGD optimizer, training steps
- **Control Nodes**: Context management, mode setting, epoch tracker
- **Visualization Nodes**: Tensor visualization

#### RL Nodes (Category: "rl")
- **PPO Agent**: Proximal Policy Optimization implementation
- **PPO Config**: Configuration for PPO hyperparameters

#### Robotics Nodes (Category: "robotics")
- **Isaac Gym Sim**: Core physics simulator integration
- **Isaac Gym Envs**: Pre-built RL environments

#### Utility Nodes (Category: "utility")
- **Logic Nodes**: OR node for conditional logic
- **Configuration**: Balancing config and control nodes

### Export System Architecture (`export_system/`)

The export system is the project's most sophisticated feature, converting visual workflows into executable Python code:

#### Key Components
- **Graph Exporter** (`graph_exporter.py`): Converts ComfyUI JSON workflows to Python scripts
- **Node Templates** (`templates/nodes/`): Python code templates for each node type
- **Queue Framework** (`templates/base/queue_framework.py`): Async queue-based execution for real-time applications
- **Node Exporters** (`node_exporters/`): Handles code generation for specific node categories

#### Export Patterns
- **Queue Templates**: Generate async queue-based code for real-time robotics applications
- **Training Runners**: Complete training loop implementations (runner.py in exported code)

### System Components
The system has three main components:
1. **Editor/Builder UI (DNNE)**: Visual graph editor where users drag and drop nodes to create neural network architectures
2. **Export System**: Converts the visual graph into standalone Python scripts
3. **Runner**: The exported Python script entry that runs independently with NVIDIA Isaac Gym

### Data Flow
1. **Visual Design**: Users create workflows in the visual graph editor
2. **Node Graph**: System represents workflows as connected node graphs
3. **Code Generation**: Export system converts graphs to Python modules (saved as a package to `export_system/exports/{workflow_name}`)
4. **Execution**: Generated code runs independently on target platforms

## Important Development Notes

### **Base Class Design Principles**
- **No Default Guessing**: Base classes should never implement "guessed" default values when subclasses forget to implement required methods. This creates hard-to-debug issues where the wrong behavior is silently used instead of failing fast.
- **Fail Fast with NotImplementedError**: When a base class method requires subclass implementation, throw `NotImplementedError` with a clear message about what needs to be implemented.
- **Example**: Instead of `return ["input"]` as a default for `get_input_names()`, throw `NotImplementedError(f"Subclass {cls.__name__} must implement get_input_names() method")`
- **Benefits**: Immediate feedback when methods are missing, prevents silent wrong behavior, makes debugging much easier

### Node Implementation Patterns
- All custom nodes inherit from base classes in their respective modules
- Nodes must provide both UI execution and export template generation
- Context used by nodes is now implicit (global) - no explicit context connections needed

### Export System Guidelines
- Each node type requires a queue template only (non-async templates are obsoleted)
- Templates use string formatting for parameter injection
- Generated code follows queue-based reactive patterns for robotics applications
- All exports include proper import management and error handling
- Export functionality accessible via "Export" button (renamed from "Run")
- All node communication uses async queue-based design similar to ROS (Robot Operating System)

### Testing Approach
- Tests driven by script `dnne-test`
- Export system tests verify code generation and execution
- Integration tests use example workflows like "MNIST Test.json"
- Queue-based tests validate real-time execution patterns

### **CRITICAL TESTING RULE**
**NEVER mark a test as complete unless it actually runs successfully.**
- If export fails → test is INCOMPLETE, not complete
- If generated code crashes → test is FAILED, not complete
- If functionality is missing → test is PENDING, not complete
- Only mark tests complete when they execute successfully from start to finish
- Document failures honestly - partial success is not success

### **CRITICAL FILE ORGANIZATION RULE**
- NEVER create ANY files in the project root directory (/home/asantanna/DNNE/DNNE-UI/) unless directed to do so. ⚠️**
- **ALL TEST FILES MUST GO TO**: `dnne-test-suite` directories ONLY

### File Structure Conventions
- Queue templates end with `_queue.py` for async execution
- Node implementations use `*_visnode.py` naming pattern in `custom_nodes/`
- Node exporters are categorized in `node_exporters/` by type (ml_nodes.py, rl_nodes.py, etc.)
- Generated code follows consistent naming and structure patterns

## Key Dependencies

### Core ML/Robotics Stack
- **PyTorch**: Deep learning framework (≥2.0.0)
- **torchvision**: Computer vision utilities
- **numpy**: Numerical computing
- **Python**: 3.10+ with async/await support

### Target Runtime
- **NVIDIA Isaac Gym**: Primary target for robotics simulation (installed at `~/isaacgym`)
- **IsaacGymEnvs**: Reinforcement learning environments (installed at `~/IsaacGymEnvs`)
- **WSL2**: Development environment support
- **GPU Cloud Providers**: Lambda and similar platforms

### ComfyUI Infrastructure
- **websockets**: Real-time communication
- **aiohttp/aiofiles**: Async web server components
- **safetensors**: Safe model serialization

### Development Tools
- Standard Python testing with export system validation
- Visual workflow testing through ComfyUI interface
- Queue-based execution testing for robotics scenarios

## Workflow Examples

The `user/default/workflows/MNIST Test.json` provides a complete example showing:
- MNIST dataset loading and batch sampling
- Two-layer neural network with ReLU activation
- Cross-entropy loss calculation
- SGD optimizer configuration
- Training step execution

This workflow demonstrates the full ML pipeline from data loading through training, and serves as a reference for implementing similar patterns.

## Technical Context

### Architecture Overview
- Built on ComfyUI's architecture but heavily modified
- Uses Python 3.10+ with async/await for modern async programming
- Target runtime is NVIDIA Isaac Gym for robotics simulation
- Frontend uses Vue.js instead of ComfyUI's vanilla JavaScript
- All node communication is async queue-based for real-time performance

### Export System Design
The export system generates clean, executable Python code that correctly implements the visual graph's logic while maintaining the async queue-based architecture needed for real-time robotics applications. Key features include:
- Queue-based templates for all nodes
- Proper variable naming (e.g., `node_10` format)
- Correct parameter processing in templates
- Full template variable substitution during export
- Exports save to `export_system/exports/{workflow_name}/runner.py` and potentially other files in that directory

### Current Capabilities (As of June 2025)
- **Fully Functional Export System**: Export functionality via "Export" button working correctly
- **Complete ML Node System**: LinearLayer, MNISTDataset, BatchSampler, Network, SGDOptimizer, CrossEntropyLoss, TrainingStep, EpochTracker, GetBatch nodes all implemented
- **Queue-Based Async Architecture**: All nodes use async queue-based execution for real-time performance
- **Network Node Pattern**: Network nodes consolidate multiple LinearLayer nodes into sequential PyTorch models
- **Training Progress Display**: EpochTracker and enhanced loss nodes provide comprehensive training statistics
- **Slot Mapping Resolution**: Fixed ComfyUI slot corruption issue with JSON-based workaround
- **Template System**: Complete template-based code generation with proper variable substitution
- **MNIST Classification Pipeline**: Full working example achieving standard ML performance benchmarks

### Export System Architecture Details
- **Graph Exporter** (`graph_exporter.py`): Core export logic with slot corruption workaround via `_fix_corrupted_slots()` method
- **Node Templates** (`templates/nodes/*_queue.py`): Queue-based templates for all node types
- **Node Exporters** (`node_exporters/ml_nodes.py`): Handles parameter extraction and template variable preparation
- **Queue Framework**: Complete async queue framework with SensorNode, QueueNode base classes and GraphRunner orchestration
- **Connection System**: Robust connection mapping that survives ComfyUI pipeline processing

## Workflow
- **Future Features** (`docs-dnne/future/`): When you have ideas for future features or improvements, create a new markdown file in the appropriate subdirectory. Keep filenames short but descriptive. Update the README.md index when adding new features. Each feature file should include: Priority (High/Medium/Low), Description, Motivation, Implementation Notes, Dependencies, and Estimated Effort.
