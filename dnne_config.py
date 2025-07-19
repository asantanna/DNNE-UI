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
            
            # Validate paths exist
            self._validate_paths()
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
    
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
                
                return re.sub(pattern, replacer, value)
            elif isinstance(value, dict):
                return {k: substitute_in_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_in_value(item) for item in value]
            return value
        
        self._config = substitute_in_value(self._config)
    
    def _validate_paths(self):
        """Validate that critical paths exist"""
        critical_paths = [
            'paths.dnne_root',
            'paths.linux_support'
        ]
        
        warnings = []
        for path_key in critical_paths:
            path_value = self.get(path_key)
            if path_value and not os.path.exists(path_value):
                warnings.append(f"Warning: Path '{path_key}' does not exist: {path_value}")
        
        if warnings:
            for warning in warnings:
                print(f"⚠️  {warning}")
    
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
        """Get a path from the paths section"""
        return self.get(f'paths.{path_key}', '')
    
    def get_conda_activate_command(self) -> str:
        """Get the conda activation command"""
        conda_path = self.get_path('conda_path')
        conda_env = self.get_path('conda_env')
        
        if conda_path and conda_env:
            return f"source {conda_path}/bin/activate {conda_env}"
        return ""
    
    def get_export_path(self, workflow_name: Optional[str] = None) -> Path:
        """Get the export path for a workflow"""
        dnne_root = Path(self.get_path('dnne_root'))
        export_base = self.get('export.export_base', 'export_system/exports')
        
        if workflow_name:
            return dnne_root / export_base / workflow_name
        else:
            default_workflow = self.get('export.default_workflow', 'Cartpole_PPO')
            return dnne_root / export_base / default_workflow
    
    def get_workflow_path(self, workflow_name: str) -> Path:
        """Get the path to a workflow JSON file"""
        dnne_root = Path(self.get_path('dnne_root'))
        workflow_path = self.get('export.workflow_path', 'user/default/workflows')
        return dnne_root / workflow_path / f"{workflow_name}.json"
    
    def get_temp_dir(self) -> Path:
        """Get temporary directory for profiling outputs"""
        return Path(self.get('profiling.temp_directory', '/tmp'))
    
    def get_all_paths(self) -> Dict[str, str]:
        """Get all configured paths"""
        return self.get('paths', {})
    
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
    return Path(config.get_path('isaac_gym_envs'))


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