"""Configuration utilities for loading YAML configs and managing timestamps."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_timestamp() -> str:
    """Get current timestamp in ISO format.
    
    Returns:
        Current timestamp as ISO string
    """
    return datetime.now().isoformat()


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data as JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path where to save the JSON file
        
    Raises:
        OSError: If file cannot be written
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary containing the loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON parsing fails
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object of the directory
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path to the project root
    """
    # Look for pyproject.toml to identify project root
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    
    # Fallback to current working directory
    return Path.cwd()
