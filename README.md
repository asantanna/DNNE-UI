# DNNE - Distributed Neural Network Editor

DNNE is a visual programming environment for building neural networks and robotics control systems. It transforms ComfyUI's visual node-based interface from a diffusion model platform into a comprehensive ML/robotics development environment with code export capabilities.

## Key Features

- **Visual Neural Network Design**: Drag and drop interface for building ML models
- **Robotics Integration**: Native support for NVIDIA Isaac Gym environments
- **Code Export**: Convert visual workflows to standalone Python scripts
- **Real-time Training**: Queue-based async architecture for robotics applications
- **Production Ready**: Export to run on cloud GPU providers or local machines

## Quick Start

### 1. Clone the Repositories

```bash
git clone https://github.com/asantanna/DNNE-UI.git
git clone https://github.com/asantanna/DNNE-UI-Frontend.git
git clone https://github.com/asantanna/DNNE-LINUX-SUPPORT.git
cd DNNE-UI
```

### 2. Configure for Your System

DNNE uses a flexible configuration system. For a typical setup where repositories are in `/workspace`:

```bash
# Run the setup script
./setup_dnne.sh /workspace

# Or manually create configuration
mkdir -p ~/.dnne
cp dnne_config.example.json ~/.dnne/config.json
# Edit ~/.dnne/config.json with your paths
```

See [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) for detailed configuration instructions.

### 3. Install Dependencies

```bash
# Create conda environment
conda create -n DNNE_PY38 python=3.8
conda activate DNNE_PY38

# Install core requirements
pip install -r requirements.txt

# For Isaac Gym/RL support (optional)
pip install -r requirements-robotics.txt

# For development (optional)
pip install -r requirements-dev.txt
```

Note: Isaac Gym, IsaacGymEnvs, and rl_games_dnne require manual installation. See [CLAUDE.md](CLAUDE.md) for details.

### 4. Run Tests

```bash
./dnne_test full
```

### 5. Start the Server (WINDOWS ONLY)

```bash
python main.py
```

Then open http://localhost:8188 in your browser.

## Project Structure

```
DNNE-UI/
├── custom_nodes/          # ML and robotics node implementations
│   ├── ml_nodes/         # Machine learning nodes
│   └── robotics_nodes/   # Isaac Gym integration nodes
├── export_system/         # Code export functionality
│   ├── templates/        # Code generation templates
│   └── exports/          # Generated Python scripts
├── dnne_test_suite/      # Comprehensive test suite
└── claude_scripts/       # Utility and development scripts
```

## Configuration

DNNE can be configured for different environments:

- **Development**: Local machine with custom paths
- **Docker/Containers**: Containerized deployments
- **HPC/Clusters**: High-performance computing environments
- **Cloud**: GPU cloud providers (Lambda, AWS, etc.)

Configuration options:
1. Environment variables (highest priority)
2. User config: `~/.dnne/config.json`
3. Project config: `dnne_config.json`

See [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) for complete details.

## Example Workflows

DNNE includes example workflows in `user/default/workflows/`:

- **MNIST Training**: Complete supervised learning pipeline
- **Cartpole PPO**: Reinforcement learning with Isaac Gym
- **Custom Networks**: Build your own architectures

## Export System

The export system converts visual workflows to standalone Python code:

```bash
# Export via UI: Click "Export" button in the interface

# Exported code structure:
export_system/exports/YourWorkflow/
├── runner.py          # Main execution script
├── nodes/             # Generated node implementations
└── framework/         # Queue-based execution framework
```

Run exported code:
```bash
cd export_system/exports/YourWorkflow
python runner.py --help
```

## Development

See [CLAUDE.md](CLAUDE.md) for detailed development documentation, including:
- Architecture overview
- Node implementation guide
- Export system details
- Testing approach

## Testing

Run the comprehensive test suite:

```bash
# Full test suite
./dnne_test full

# Unit tests only
./dnne_test unit

# Integration tests only
./dnne_test integration

# With coverage report
./dnne_test coverage
```

## Related Repositories

- **Frontend**: [DNNE-UI-Frontend](https://github.com/asantanna/DNNE-UI-Frontend) - Vue.js-based visual editor
- **Linux Support**: [DNNE-LINUX-SUPPORT](https://github.com/asantanna/DNNE-LINUX-SUPPORT) - Isaac Gym and dependencies

## Original ComfyUI

DNNE is built on top of ComfyUI. See [COMFYUI-README.md](COMFYUI-README.md) for the original ComfyUI documentation.

## License

[License information here]

## Contributing

[Contributing guidelines here]