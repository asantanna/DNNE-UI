#!/usr/bin/env python3
"""
DNNE Configuration Module

Provides centralized configuration management for DNNE paths and settings.
Supports loading from JSON files with environment variable overrides.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


class DNNEConfig:
    """Manages DNNE configuration with support for multiple config sources"""
    
    def __init__(self):
        self._config = {}
        self._config_file = None
        self.load_config()
    
    def load_config(self):
        """Load configuration from various sources with priority:
        1. Environment variable DNNE_CONFIG_PATH
        2. User home directory ~/.dnne/config.json
        3. Project root dnne_config.json
        """
        # Try environment variable first
        env_config_path = os.environ.get('DNNE_CONFIG_PATH')
        if env_config_path and os.path.exists(env_config_path):
            self._config_file = env_config_path
            self._load_from_file(env_config_path)
            return
        
        # Try user home directory
        home_config = Path.home() / '.dnne' / 'config.json'
        if home_config.exists():
            self._config_file = str(home_config)
            self._load_from_file(home_config)
            return
        
        # Try project root (where this file is located)
        project_config = Path(__file__).parent / 'dnne_config.json'
        if project_config.exists():
            self._config_file = str(project_config)
            self._load_from_file(project_config)
            return
        
        # If no config found, raise error
        raise FileNotFoundError(
            "No DNNE configuration file found. Please create dnne_config.json "
            "in the project root or set DNNE_CONFIG_PATH environment variable."
        )
    
    def _load_from_file(self, config_path: Path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                self._config = json.load(f)
            
            # Apply environment variable substitutions
            self._substitute_env_vars()
            
            # Validate paths exist after loading
            self._validate_paths()
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
    
    def _convert_path_for_os(self, path: str) -> str:
        """Convert path based on current OS"""
        import platform
        import re
        
        # Don't try to convert Windows paths on Linux
        if platform.system() != "Windows":
            # Check for Windows path pattern: C:\ or C:/ or \\server\share
            windows_path_pattern = r'^[A-Za-z]:[\\\/]|^\\\\[^\\]+'
            if re.match(windows_path_pattern, path):
                return path  # Return as-is, will fail validation appropriately
        
        # First expand ~ if present
        if path.startswith('~'):
            if platform.system() == "Windows":
                # On Windows, ~ means the WSL home
                wsl_prefix = self.get('dnne.wsl.filesystem_prefix', self.get('wsl.windows_prefix', '\\\\wsl.localhost\\Ubuntu'))
                home_path = self.get('dnne.wsl.home_path', self.get('wsl.home_path', '/home/asantanna'))
                path = path.replace('~', wsl_prefix + home_path)
            else:
                # On Linux, use normal expansion
                path = os.path.expanduser(path)
        
        return path
    
    def _substitute_env_vars(self):
        """Replace ${VAR} patterns with environment variable values"""
        def substitute_in_value(value):
            if isinstance(value, str):
                # Replace ${VAR} with environment variable
                import re
                pattern = r'\$\{([^}]+)\}'
                
                def replacer(match):
                    var_name = match.group(1)
                    return os.environ.get(var_name, match.group(0))
                
                value = re.sub(pattern, replacer, value)
                return value
            elif isinstance(value, dict):
                return {k: substitute_in_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_in_value(item) for item in value]
            return value
        
        self._config = substitute_in_value(self._config)
    
    def _validate_paths(self):
        """Validate that critical paths exist"""
        import platform
        
        # Skip validation on Windows for WSL paths
        if platform.system() == "Windows":
            # print("Note: Path validation skipped on Windows (WSL paths cannot be validated from Windows Python)")
            return
            
        # On Linux, only validate Linux paths
        critical_paths = []
        
        # Check which paths to validate based on config structure
        if 'exported' in self._config and 'paths' in self._config['exported']:
            # New structure - validate exported paths
            for key in ['linux_support']:
                if key in self._config['exported']['paths']:
                    critical_paths.append(('exported.paths.' + key, key))
        elif 'paths' in self._config:
            # Old structure - validate appropriate paths
            for key in ['linux_support']:
                if key in self._config['paths']:
                    critical_paths.append(('paths.' + key, key))
        
        errors = []
        for path_config, path_name in critical_paths:
            # Get raw path value
            path_value = self.get(path_config)
            if path_value:
                # Expand and check
                expanded_path = os.path.expanduser(path_value)
                if not os.path.exists(expanded_path):
                    errors.append(f"Critical path '{path_name}' does not exist: {expanded_path}")
        
        if errors:
            for error in errors:
                print(f"❌ {error}")
            raise FileNotFoundError(
                "Critical paths are missing. Please check your dnne_config.json and ensure all paths exist."
            )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'paths.dnne_root')"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_path(self, path_key: str) -> str:
        """Get a path from the paths section, converted for current OS"""
        import platform
        
        # On Windows, try dnne.paths first
        if platform.system() == "Windows":
            raw_path = self.get(f'dnne.paths.{path_key}', '')
            if raw_path:
                return self._convert_path_for_os(raw_path)
        
        # On Linux or if not found in dnne.paths, try exported.paths
        raw_path = self.get(f'exported.paths.{path_key}', '')
        if not raw_path:
            # Fall back to old structure
            raw_path = self.get(f'paths.{path_key}', '')
        
        return self._convert_path_for_os(raw_path)
    
    def get_conda_activate_command(self) -> str:
        """Get the conda activation command"""
        # Try exported conda config first (for Linux)
        conda_path = self.get('exported.conda.conda_path', '')
        conda_env = self.get('exported.conda.conda_env', '')
        
        # Fall back to old structure
        if not conda_path:
            conda_path = self.get('paths.conda_path', '')
        if not conda_env:
            conda_env = self.get('paths.conda_env', '')
        
        if conda_path and conda_env:
            conda_path = self._convert_path_for_os(conda_path)
            return f"source {conda_path}/bin/activate {conda_env}"
        return ""
    
    def get_export_path(self, workflow_name: Optional[str] = None) -> Path:
        """Get the export path for a workflow"""
        dnne_root = Path(self.get_path('dnne_root'))
        export_base = self.get('dnne.export.export_base', self.get('export.export_base', 'export_system/exports'))
        
        if workflow_name:
            return dnne_root / export_base / workflow_name
        else:
            default_workflow = self.get('dnne.export.default_workflow', self.get('export.default_workflow', 'Cartpole_PPO'))
            return dnne_root / export_base / default_workflow
    
    def get_workflow_path(self, workflow_name: str) -> Path:
        """Get the path to a workflow JSON file"""
        dnne_root = Path(self.get_path('dnne_root'))
        workflow_dir = self.get('dnne.export.workflow_dir', self.get('export.workflow_path', 'user/default/workflows'))
        return dnne_root / workflow_dir / f"{workflow_name}.json"
    
    def get_temp_dir(self) -> Path:
        """Get temporary directory for profiling outputs"""
        # Try new structure first
        temp_dir = self.get('exported.paths.temp_directory', '')
        if not temp_dir:
            temp_dir = self.get('dnne.paths.temp_directory', '')
        if not temp_dir:
            # Fall back to old structure
            temp_dir = self.get('profiling.temp_directory', '/tmp')
        return Path(temp_dir)
    
    def get_all_paths(self) -> Dict[str, str]:
        """Get all configured paths"""
        # Merge paths from all sections
        paths = {}
        paths.update(self.get('dnne.paths', {}))
        paths.update(self.get('exported.paths', {}))
        paths.update(self.get('paths', {}))  # Old structure
        return paths
    
    def __str__(self) -> str:
        """String representation showing loaded config file"""
        return f"DNNEConfig(config_file='{self._config_file}')"
    
    def __repr__(self) -> str:
        return self.__str__()


# Global configuration instance
config = DNNEConfig()


# Convenience functions for common paths
def get_dnne_root() -> Path:
    """Get DNNE root directory"""
    return Path(config.get_path('dnne_root'))


def get_isaac_gym_envs_path() -> Path:
    """Get IsaacGymEnvs directory"""
    linux_support = Path(config.get_path('linux_support'))
    subdir = config.get('shared.linux_support_subdirs.isaac_gym_envs', config.get('linux_support_subdirs.isaac_gym_envs', 'IsaacGymEnvs'))
    return linux_support / subdir


def get_isaac_gym_path() -> Path:
    """Get Isaac Gym directory"""
    linux_support = Path(config.get_path('linux_support'))
    subdir = config.get('shared.linux_support_subdirs.isaac_gym', config.get('linux_support_subdirs.isaac_gym', 'isaacgym'))
    return linux_support / subdir


def get_rl_games_path() -> Path:
    """Get rl_games_dnne directory"""
    linux_support = Path(config.get_path('linux_support'))
    subdir = config.get('shared.linux_support_subdirs.rl_games_dnne', config.get('linux_support_subdirs.rl_games_dnne', 'rl_games_dnne'))
    return linux_support / subdir


def get_linux_support_path() -> Path:
    """Get Linux support directory"""
    return Path(config.get_path('linux_support'))


def get_conda_activate() -> str:
    """Get conda activation command"""
    return config.get_conda_activate_command()


# For backwards compatibility
def get_export_dir(workflow_name: Optional[str] = None) -> Path:
    """Get export directory for a workflow"""
    return config.get_export_path(workflow_name)