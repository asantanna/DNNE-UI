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
- **Standard Templates**: Generate synchronous code for training pipelines
- **Queue Templates**: Generate async queue-based code for real-time robotics applications
- **Training Runners**: Complete training loop implementations

### System Components
The system has three main components:
1. **Builder UI (DNNE-UI)**: Visual graph editor where users drag and drop nodes to create neural network architectures
2. **Export System**: Converts the visual graph into standalone Python scripts
3. **Runner**: The exported Python script that runs independently with NVIDIA Isaac Gym

### Data Flow
1. **Visual Design**: Users create workflows in the visual graph editor
2. **Node Graph**: System represents workflows as connected node graphs
3. **Code Generation**: Export system converts graphs to Python modules (saved to `export_system/exports/{workflow_name}/runner.py`)
4. **Execution**: Generated code runs independently on target platforms with async queue-based architecture

## Important Development Notes

### Node Implementation Patterns
- All custom nodes inherit from base classes in their respective modules
- Nodes must provide both UI execution and export template generation
- Context is now implicit (global) - no explicit context connections needed

### Export System Guidelines
- Each node type requires both a standard template and queue template
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
- Templates end with `_template.py` for standard execution
- Queue variants end with `_queue.py` for async execution
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

## Known Issues and Current Development State

### What's Working
- Basic export functionality via the "Export" button (renamed from "Run")
- Node system with ML nodes (LinearLayer, MNISTDataset, Loss, etc.)
- Template-based code generation
- Exports save to `export_system/exports/{workflow_name}/runner.py`

### Current Issues
- **Template Inconsistencies**: Some nodes still use old configuration-style templates instead of queue-based templates
- **Syntax Errors in Generated Code**:
  - Variable names starting with numbers (e.g., `10_node`) - should use `node_10` instead
  - Incorrect bias parameters (`bias=relu` instead of `bias=True`)
  - Missing class definitions for some nodes
- **Missing Node Templates**: TrainingStep node needs a queue-based template
- **Template Variable Processing**: Some templates show raw template code instead of processed values

### Immediate Development Priorities
1. Convert all remaining nodes to use queue-based templates exclusively
2. Fix variable naming issue in main execution template (use `node_10` instead of `10_node`)
3. Fix parameter processing in templates (especially bias parameter in LinearLayer)
4. Ensure all template variables are properly replaced during export
5. Complete TrainingStep node queue-based template implementation

### Technical Context
- Built on ComfyUI's architecture but heavily modified
- Uses Python 3.10+ with async/await
- Target runtime is NVIDIA Isaac Gym for robotics simulation
- Frontend uses Vue.js instead of ComfyUI's vanilla JavaScript
- All node communication is async queue-based for real-time performance

The main challenge is ensuring the export system generates clean, executable Python code that correctly implements the visual graph's logic while maintaining the async queue-based architecture needed for real-time robotics applications.