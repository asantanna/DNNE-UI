# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DNNE** (Drag and Drop Neural Network Environment) is a visual programming environment for building neural networks and robotics control systems. It's based on ComfyUI's architecture but adapted for machine learning and robotics applications instead of image generation. The project transforms ComfyUI's visual node-based interface from a diffusion model platform into a comprehensive ML/robotics development environment with code export capabilities.

### Key Innovation
The primary innovation is the **export system** that converts visual node graphs into standalone, production-ready Python modules that can run on WSL2, GPU cloud providers (like Lambda), or integrate with robotics simulators like NVIDIA Isaac Gym. Unlike ComfyUI, DNNE doesn't execute graphs directly - instead, it exports them as Python scripts designed to run in tight loops with simulators.

### Repository Structure

#### Backend Repository (This Repository)
Contains the main DNNE-UI backend with:
- `server.py` - Modified ComfyUI server that handles export functionality
- `custom_nodes/` - Node implementations for ML and robotics
- `export_system/` - Export system that converts visual graphs to Python code

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
python test_generated_queue.py
python test_camera_queue.py
python test_multisensor_queue.py
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
- **Debug execution**: Use `test_debug.py` for troubleshooting

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
- **NVIDIA Isaac Gym**: Primary target for robotics simulation
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
- **TODO.md** (`TODO.md`): Whenever you change your internal to-do list, update TODO.md also. This file is meant to be a permanent record of cumulative to-do items created and completed during the lifetime of the project. This file is divided into two sections.  The first has a header "OPEN TO-DO ITEMS" followed by open todo items only.  After a line separator, a header says "COMPLETED TO-DO ITEMS" followed by a listing of all the completed todo items.  Use a strike-through font on completed items.
- **TASKS.md** (`TASKS.md`): This file is updated by the developers with high-level features and any other concerns to be addressed at a later time.
