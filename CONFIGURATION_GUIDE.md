# DNNE Configuration Guide

This guide explains how to configure DNNE (Distributed Neural Network Editor) for your specific system setup.

## Overview

DNNE uses a configuration system that allows you to customize paths and settings for your environment. The configuration can be set through:
1. Environment variables (highest priority)
2. User configuration file (`~/.dnne/config.json`)
3. Project configuration file (`dnne_config.json` in project root)

## Quick Start: Example Configuration

If you've cloned all repositories into `/workspace`, follow these steps:

### 1. Create Configuration File

Create a file at `~/.dnne/config.json` (or modify `dnne_config.json` in the project root):

```json
{
  "paths": {
    "dnne_root": "/workspace/DNNE-UI",
    "linux_support": "/workspace/DNNE-LINUX-SUPPORT",
    "isaac_gym_envs": "/workspace/IsaacGymEnvs",
    "isaac_gym": "/workspace/isaacgym",
    "rl_games_dnne": "/workspace/rl_games_dnne",
    "conda_path": "/home/your_username/miniconda3",
    "conda_env": "DNNE_PY38"
  },
  "export": {
    "default_workflow": "Cartpole_PPO",
    "workflow_path": "user/default/workflows",
    "export_base": "export_system/exports"
  },
  "profiling": {
    "default_num_envs": 512,
    "default_timeout": 300,
    "temp_directory": "/tmp"
  }
}
```

### 2. Using Environment Variables

Alternatively, you can set environment variables in your shell profile (`.bashrc`, `.zshrc`, etc.):

```bash
# DNNE Configuration
export DNNE_CONFIG_PATH="/workspace/DNNE-UI/my_config.json"  # Use custom config file
# OR set individual paths:
export DNNE_ROOT="/workspace/DNNE-UI"
export DNNE_LINUX_SUPPORT="/workspace/DNNE-LINUX-SUPPORT"
export DNNE_ISAAC_GYM_ENVS="/workspace/IsaacGymEnvs"
export DNNE_ISAAC_GYM="/workspace/isaacgym"
export DNNE_RL_GAMES_DNNE="/workspace/rl_games_dnne"
export DNNE_CONDA_PATH="/home/your_username/miniconda3"
export DNNE_CONDA_ENV="DNNE_PY38"
```

## Configuration Options

### Path Configuration

| Key | Description | Default |
|-----|-------------|---------|
| `dnne_root` | Main DNNE-UI repository path | `/mnt/e/ALS-Projects/DNNE/DNNE-UI` |
| `linux_support` | DNNE Linux support directory | `/home/asantanna/DNNE-LINUX-SUPPORT` |
| `isaac_gym_envs` | IsaacGymEnvs installation path | `{linux_support}/IsaacGymEnvs` |
| `isaac_gym` | Isaac Gym installation path | `{linux_support}/isaacgym` |
| `rl_games_dnne` | Modified rl_games path | `{linux_support}/rl_games_dnne` |
| `conda_path` | Miniconda/Anaconda installation | `/home/asantanna/miniconda` |
| `conda_env` | Conda environment name | `DNNE_PY38` |

### Export Configuration

| Key | Description | Default |
|-----|-------------|---------|
| `default_workflow` | Default workflow for testing | `Cartpole_PPO` |
| `workflow_path` | Path to workflow files | `user/default/workflows` |
| `export_base` | Base directory for exports | `export_system/exports` |

### Profiling Configuration

| Key | Description | Default |
|-----|-------------|---------|
| `default_num_envs` | Number of parallel environments | `512` |
| `default_timeout` | Default timeout in seconds | `300` |
| `temp_directory` | Temporary file directory | `/tmp` |

## Step-by-Step Setup for New Systems

### 1. Clone Required Repositories

```bash
cd /workspace  # or your preferred directory
git clone https://github.com/asantanna/DNNE-UI.git
git clone https://github.com/asantanna/DNNE-UI-Frontend.git
git clone https://github.com/asantanna/DNNE-LINUX-SUPPORT.git

# DNNE-LINUX-SUPPORT already contains:
# - isaacgym/
# - IsaacGymEnvs/
# - rl_games_dnne/
```

### 2. Create Conda Environment and Install Dependencies

```bash
conda create -n DNNE_PY38 python=3.8
conda activate DNNE_PY38
cd /workspace/DNNE-UI
pip install -r requirements.txt

# Isaac Gym components in DNNE-LINUX-SUPPORT are used directly via sys.path
# No installation needed - the code adds these paths dynamically
# Note: This currently uses hardcoded paths that need to be updated to use the configuration system
```

### 3. Configure DNNE

Create `~/.dnne/config.json`:

```bash
mkdir -p ~/.dnne
cat > ~/.dnne/config.json << 'EOF'
{
  "paths": {
    "dnne_root": "/workspace/DNNE-UI",
    "linux_support": "/workspace/DNNE-LINUX-SUPPORT",
    "isaac_gym_envs": "/workspace/DNNE-LINUX-SUPPORT/IsaacGymEnvs",
    "isaac_gym": "/workspace/DNNE-LINUX-SUPPORT/isaacgym",
    "rl_games_dnne": "/workspace/DNNE-LINUX-SUPPORT/rl_games_dnne",
    "conda_path": "/path/to/your/conda",
    "conda_env": "DNNE_PY38"
  }
}
EOF
```

### 4. Test Configuration

```bash
cd /workspace/DNNE-UI
python -c "from dnne_config import DNNEConfig; c = DNNEConfig(); print(c.get_all())"
```

This should display your configuration settings.

### 5. Run Tests

```bash
./dnne-test full
```

## Common Configuration Scenarios

### Scenario 1: Docker/Container Setup

```json
{
  "paths": {
    "dnne_root": "/app/dnne",
    "linux_support": "/app/linux-support",
    "isaac_gym_envs": "/app/linux-support/IsaacGymEnvs",
    "isaac_gym": "/app/linux-support/isaacgym",
    "rl_games_dnne": "/app/linux-support/rl_games_dnne",
    "conda_path": "/opt/conda",
    "conda_env": "base"
  }
}
```

### Scenario 2: HPC/Cluster Environment

```json
{
  "paths": {
    "dnne_root": "/home/username/projects/DNNE-UI",
    "linux_support": "/home/username/libs/DNNE-LINUX-SUPPORT",
    "isaac_gym_envs": "/home/username/libs/IsaacGymEnvs",
    "isaac_gym": "/home/username/libs/isaacgym",
    "rl_games_dnne": "/home/username/libs/rl_games_dnne",
    "conda_path": "/apps/conda/2023.03",
    "conda_env": "dnne-env"
  },
  "profiling": {
    "default_num_envs": 1024,
    "temp_directory": "/scratch/username/tmp"
  }
}
```

### Scenario 3: Windows WSL2

```json
{
  "paths": {
    "dnne_root": "/mnt/c/Users/YourName/Projects/DNNE-UI",
    "linux_support": "/home/yourname/DNNE-LINUX-SUPPORT",
    "isaac_gym_envs": "/home/yourname/DNNE-LINUX-SUPPORT/IsaacGymEnvs",
    "isaac_gym": "/home/yourname/DNNE-LINUX-SUPPORT/isaacgym",
    "rl_games_dnne": "/home/yourname/DNNE-LINUX-SUPPORT/rl_games_dnne",
    "conda_path": "/home/yourname/miniconda3",
    "conda_env": "DNNE_PY38"
  }
}
```

## Configuration Priority

The configuration system loads settings in this order (later sources override earlier ones):

1. Default configuration (hardcoded in `dnne_config.py`)
2. Project `dnne_config.json` file
3. User home directory `~/.dnne/config.json`
4. File specified by `DNNE_CONFIG_PATH` environment variable
5. Individual environment variables (e.g., `DNNE_ROOT`, `DNNE_ISAAC_GYM`)

## Troubleshooting

### Issue: "Configuration file not found"
- **Solution**: Create `~/.dnne/config.json` or set `DNNE_CONFIG_PATH` environment variable

### Issue: "Isaac Gym not found"
- **Solution**: Verify `isaac_gym` path in configuration points to the correct installation
- **Check**: `ls -la /your/path/to/isaacgym/python/isaacgym`

### Issue: "Conda environment not found"
- **Solution**: Update `conda_path` and `conda_env` in configuration
- **Verify**: `source /your/conda/path/bin/activate YOUR_ENV_NAME`

### Issue: Tests fail with path errors
- **Solution**: Run configuration test to verify all paths:
  ```python
  from dnne_config import DNNEConfig
  config = DNNEConfig()
  
  # Check all paths exist
  import os
  for key, path in config.get('paths').items():
      exists = os.path.exists(path)
      print(f"{key}: {path} - {'✓' if exists else '✗ NOT FOUND'}")
  ```

## Advanced Configuration

### Using Multiple Configurations

You can maintain multiple configuration files for different environments:

```bash
# Development
export DNNE_CONFIG_PATH=~/.dnne/config-dev.json

# Production
export DNNE_CONFIG_PATH=~/.dnne/config-prod.json

# Testing
export DNNE_CONFIG_PATH=~/.dnne/config-test.json
```

### Programmatic Configuration

You can also configure DNNE programmatically:

```python
from dnne_config import DNNEConfig

# Create custom configuration
config = DNNEConfig()
config.set('paths.dnne_root', '/custom/path/to/dnne')
config.set('profiling.default_num_envs', 1024)

# Use in your code
dnne_root = config.get('paths.dnne_root')
```

## Configuration for CI/CD

For automated testing and deployment:

```yaml
# Example GitHub Actions configuration
env:
  DNNE_ROOT: ${{ github.workspace }}/DNNE-UI
  DNNE_LINUX_SUPPORT: ${{ github.workspace }}/DNNE-LINUX-SUPPORT
  DNNE_ISAAC_GYM: ${{ github.workspace }}/DNNE-LINUX-SUPPORT/isaacgym
  DNNE_CONDA_ENV: base
```

## Getting Help

If you encounter configuration issues:

1. Check the configuration test output
2. Verify all paths exist and are accessible
3. Ensure conda environment is properly set up
4. Check file permissions (especially for Isaac Gym)
5. Report issues at https://github.com/anthropics/claude-code/issues

## Example: Complete Setup Script

Here's a complete setup script for a new system:

```bash
#!/bin/bash
# setup_dnne.sh - Complete DNNE setup for /workspace directory

# Set base directory
WORKSPACE="/workspace"

# Clone repositories
cd $WORKSPACE
git clone https://github.com/asantanna/DNNE-UI.git
git clone https://github.com/asantanna/DNNE-UI-Frontend.git
git clone https://github.com/asantanna/DNNE-LINUX-SUPPORT.git
# DNNE-LINUX-SUPPORT already contains isaacgym/, IsaacGymEnvs/, and rl_games_dnne/

# Create conda environment
conda create -n DNNE_PY38 python=3.8 -y
conda activate DNNE_PY38

# Install requirements
cd $WORKSPACE/DNNE-UI
pip install -r requirements.txt

# Create configuration
mkdir -p ~/.dnne
cat > ~/.dnne/config.json << EOF
{
  "paths": {
    "dnne_root": "$WORKSPACE/DNNE-UI",
    "linux_support": "$WORKSPACE/DNNE-LINUX-SUPPORT",
    "isaac_gym_envs": "$WORKSPACE/DNNE-LINUX-SUPPORT/IsaacGymEnvs",
    "isaac_gym": "$WORKSPACE/DNNE-LINUX-SUPPORT/isaacgym",
    "rl_games_dnne": "$WORKSPACE/DNNE-LINUX-SUPPORT/rl_games_dnne",
    "conda_path": "$(dirname $(dirname $(which conda)))",
    "conda_env": "DNNE_PY38"
  }
}
EOF

# Test configuration
python -c "from dnne_config import DNNEConfig; c = DNNEConfig(); print('Configuration loaded successfully!')"

# Run tests
./dnne-test full

echo "DNNE setup complete!"
```

Save this script and run with: `bash setup_dnne.sh`