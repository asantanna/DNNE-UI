# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DNNE** (Drag and Drop Neural Network Environment) is a visual programming environment for building neural networks and robotics control systems. It's based on ComfyUI's architecture but adapted for machine learning and robotics applications instead of image generation. The project transforms ComfyUI's visual node-based interface from a diffusion model platform into a comprehensive ML/robotics development environment with code export capabilities.

### Key Innovation
The primary innovation is the **export system** that converts visual node graphs into standalone, production-ready Python modules that can run on WSL2, GPU cloud providers (like Lambda), or integrate with robotics simulators like NVIDIA Isaac Gym. Unlike ComfyUI, DNNE doesn't execute graphs directly - instead, it exports them as Python scripts designed to run in tight loops with simulators.

### Repository Structure

#### Code Locations
- Backend code is checked out to: `/mnt/e/ALS-Projects/DNNE/DNNE-UI`
- Front end code is checked out to: `/mnt/e/ALS-Projects/DNNE/DNNE-UI-Frontend`

#### Backend Repository (This Repository)
Contains the main DNNE-UI backend with:
- `server.py` - Modified ComfyUI server that handles export functionality
- `custom_nodes/` - Node implementations for ML and robotics
- `export_system/` - Export system that converts visual graphs to Python code
- `claude_scripts/` - Claude-created utility and test scripts for development

#### Frontend Repository
**GitHub**: https://github.com/asantanna/DNNE-UI-Frontend.git
Vue.js-based frontend providing the visual graph editor interface (replaces original ComfyUI frontend).

## Development Commands

### Environment Setup
The project requires a properly configured conda environment with PyTorch. To activate it:
```bash
source /home/asantanna/miniconda/bin/activate DNNE_PY38
```

**Note**: The standard `conda activate DNNE_PY38` command may not work in all shell contexts. Use the full path activation method above for reliable environment activation.

If the conda environment is not activated, you may encounter errors like:
- `ModuleNotFoundError: No module named 'torch'`
- Issues with CUDA/GPU detection
- Missing dependencies that are installed in the conda environment

### Isaac Gym Integration
IsaacGym and IsaacGymEnvs are installed and verified working:
- **IsaacGym**: `~/isaacgym` - Core physics simulation library
- **IsaacGymEnvs**: `~/IsaacGymEnvs` - Pre-built reinforcement learning environments
- **Import Order**: Always import `isaacgym` before `torch` to avoid conflicts
- **GPU Support**: Verified working with CUDA and GPU PhysX acceleration
- **Environment Testing**: Cartpole and other environments tested successfully

**⚠️ CRITICAL ISAAC GYM IMPORT ORDER FIX ⚠️**
The export system MUST ensure Isaac Gym nodes are imported before any torch-using nodes in `nodes/__init__.py`. This is handled in `graph_exporter._generate_node_init()` which sorts Isaac Gym nodes first. Without this, you get "PyTorch was imported before isaacgym" errors.

### Starting the Application
```bash
python main.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Testing Export System
```bash
python export_system/test_export.py
python export_system/test_exporter_linear.py
python claude_scripts/test_modular_export.py
python claude_scripts/test_modular_run.py
python claude_scripts/debug_runner.py
python claude_scripts/benchmark_pytorch_direct.py
```

### Running Exported Scripts
After exporting a workflow, run the generated script:
```bash
cd export_system/exports/{workflow_name}
python runner.py
```
Note: Ensure the conda environment is activated before running exported scripts.

### Common Development Tasks
- **Export workflows to Python**: Use the export system via the UI or programmatically through `export_system/graph_exporter.py`
- **Add new node types**: Implement in `custom_nodes/ml_nodes/` or `custom_nodes/robotics_nodes/`
- **Test node templates**: Use the test files in `export_system/`
- **Debug execution**: Use scripts in `claude_scripts/` for troubleshooting and testing
- **Benchmark performance**: Use `claude_scripts/benchmark_pytorch_direct.py` for performance comparisons

## Architecture Overview

### Core System Structure
- **Entry Point**: `main.py` - Initializes ComfyUI with ML/robotics extensions
- **Node System**: `nodes.py` - Base DNNE node classes and robotics type integration
- **Execution**: `execution.py` - Minimal execution engine optimized for robotics workflows
- **Server**: `server.py` - Web API and interface

### Custom Node Categories

#### ML Nodes (`custom_nodes/ml_nodes/`)
- **Data Nodes**: MNIST dataset, batch sampling, data loading
- **Layer Nodes**: Linear layers, Conv2D, activation functions, dropout, batch normalization
- **Training Nodes**: Cross-entropy loss, accuracy metrics, SGD optimizer, training steps
- **Control Nodes**: Context management, mode setting
- **Visualization Nodes**: Tensor visualization

#### Robotics Nodes (`custom_nodes/robotics_nodes/`)
- **Sensor Nodes**: IMU, camera, and other sensor integrations
- **Control Nodes**: Robot controller and manipulation systems
- **Network Nodes**: Decision, vision, and sound processing networks

### Export System Architecture (`export_system/`)

The export system is the project's most sophisticated feature, converting visual workflows into executable Python code:

#### Key Components
- **Graph Exporter** (`graph_exporter.py`): Converts ComfyUI JSON workflows to Python scripts
- **Node Templates** (`templates/nodes/`): Python code templates for each node type
- **Queue Framework** (`templates/base/queue_framework.py`): Async queue-based execution for real-time applications
- **Node Exporters** (`node_exporters/`): Handles code generation for specific node categories

#### Export Patterns
- **Queue Templates**: Generate async queue-based code for real-time robotics applications
- **Training Runners**: Complete training loop implementations

### System Components
The system has three main components:
1. **Builder UI (DNNE-UI)**: Visual graph editor where users drag and drop nodes to create neural network architectures
2. **Export System**: Converts the visual graph into standalone Python scripts
3. **Runner**: The exported Python script entry that runs independently with NVIDIA Isaac Gym

### Data Flow
1. **Visual Design**: Users create workflows in the visual graph editor
2. **Node Graph**: System represents workflows as connected node graphs
3. **Code Generation**: Export system converts graphs to Python modules (saved to `export_system/exports/{workflow_name}/runner.py and potentially other files in that directory`)
4. **Execution**: Generated code runs independently on target platforms with async queue-based architecture

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
- Each node type requires a queue template only (standard templates are obsoleted)
- Templates use string formatting for parameter injection
- Generated code follows queue-based reactive patterns for robotics applications
- All exports include proper import management and error handling
- Export functionality accessible via "Export" button (renamed from "Run")
- All node communication uses async queue-based design similar to ROS (Robot Operating System)

### Testing Approach
- Unit tests in `tests-unit/` directory
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

### **⚠️ CRITICAL SILENT FAILURE PATTERN ⚠️**
**INFERENCE MODE SILENT FAILURE**: Tests often pass with "✅ All tests passed!" but inference does NOTHING.
**SYMPTOMS**: Training works, inference "completes" but shows 0 computations and no accuracy.
**ROOT CAUSE**: Training triggers disabled in inference mode, so no data flows through network.
**DETECTION**: Always check inference logs for "0 computations" - this means NO inference happened.
**SOLUTION**: ✅ FIXED - GetBatch template updated with auto-trigger mechanism for inference mode.

### **⚠️ CHECKPOINT LOADING ACCURACY DROP PATTERN ⚠️**
**SYMPTOMS**: Training accuracy ~90%, inference accuracy drops to ~8% (random chance levels).
**DETECTION**: Large accuracy gap between training and inference despite successful checkpoint loading.
**POSSIBLE CAUSES**: Model state not properly restored, device mismatch, evaluation on wrong dataset.
**STATUS**: Under investigation - need to verify checkpoint loading integrity in inference mode.

### **CRITICAL FILE ORGANIZATION RULE**
**⚠️ ABSOLUTE PROHIBITION: NEVER create ANY files in the project root directory (/mnt/e/ALS-Projects/DNNE/DNNE-UI/) ⚠️**

**EXPORTS MUST GO TO**: `export_system/exports/{workflow_name}/` ONLY
**TEST FILES MUST GO TO**: `claude_scripts/` or `tests-dnne/` directories ONLY

**REPEATED VIOLATIONS**: Creating files in project root has happened multiple times. 
**ALWAYS double-check export paths before running ANY export command.**
**The export system default behavior exports to project root - you MUST override this.**
- **Project root**: Keep clean of temporary/test files
- Before creating ANY file, verify you're in the correct directory

### File Structure Conventions
- Queue templates end with `_queue.py` for async execution
- Node exporters mirror the custom_nodes directory structure
- Generated code follows consistent naming and structure patterns

### Migration Context
The project recently underwent a context removal migration:
- Removed explicit context connections from UI
- Made context implicit in generated code
- Simplified visual workflows while maintaining functionality
- Backup files with context logic preserved for reference

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

### Recent Major Achievements (June 2025)
1. **Slot Mapping Fix**: Resolved critical issue where ComfyUI pipeline corrupted all `to_slot` values to 0, implemented JSON-based workaround that reads original workflow to restore correct connections
2. **Network Node Implementation**: Successfully implemented Network node pattern that consolidates multiple LinearLayer nodes into unified PyTorch Sequential models
3. **Training Progress System**: Added EpochTracker, enhanced GetBatch and CrossEntropyLoss nodes to provide real-time training statistics and epoch summaries
4. **MNIST Optimization**: Implemented and tested optimized MNIST training with proper learning rate (0.1), momentum (0.9), achieving expected training behavior
5. **Device Compatibility**: Fixed GPU/CPU device mismatch issues in loss computation templates
6. **Export Registration**: Completed node exporter registration system for all ML nodes

### Export System Architecture Details
- **Graph Exporter** (`graph_exporter.py`): Core export logic with slot corruption workaround via `_fix_corrupted_slots()` method
- **Node Templates** (`templates/nodes/*_queue.py`): Queue-based templates for all node types
- **Node Exporters** (`node_exporters/ml_nodes.py`): Handles parameter extraction and template variable preparation
- **Queue Framework**: Complete async queue framework with SensorNode, QueueNode base classes and GraphRunner orchestration
- **Connection System**: Robust connection mapping that survives ComfyUI pipeline processing

## Workflow
- **Future Features** (`docs-dnne/future/`): When you have ideas for future features or improvements, create a new markdown file in the appropriate subdirectory. Keep filenames short but descriptive. Update the README.md index when adding new features. Each feature file should include: Priority (High/Medium/Low), Description, Motivation, Implementation Notes, Dependencies, and Estimated Effort.
