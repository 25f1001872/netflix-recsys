"""
src/utils/paths.py
------------------
Path resolution utilities for robust cross-platform path handling.

Ensures scripts work regardless of where they're executed from.
Resolves all paths relative to the project root.
"""

import os
from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Works by finding the directory containing configs/config.yaml.
    Falls back to parent of scripts/ if called from there.
    
    Returns:
        Path: Absolute path to project root
    """
    # Try to find config.yaml
    current = Path(__file__).parent.parent.parent  # src/utils -> src -> project root
    if (current / "configs" / "config.yaml").exists():
        return current
    
    # Fallback: try current working directory
    cwd = Path.cwd()
    if (cwd / "configs" / "config.yaml").exists():
        return cwd
    
    # Last resort: parent of parent
    return current


def resolve_path(path_spec: Union[str, Path], make_absolute: bool = True) -> Path:
    """
    Resolve a path specification to an absolute Path.
    
    Args:
        path_spec: Relative or absolute path string/Path
        make_absolute: If True, return absolute path; else return relative
    
    Returns:
        Path: Resolved absolute path (or relative if make_absolute=False)
        
    Examples:
        >>> resolve_path("data/raw")
        PosixPath('/home/user/netflix-recsys/data/raw')
        
        >>> resolve_path("/tmp/models")
        PosixPath('/tmp/models')
    """
    path = Path(path_spec)
    
    # Already absolute
    if path.is_absolute():
        return path
    
    # Relative path: resolve from project root
    project_root = get_project_root()
    resolved = project_root / path
    
    return resolved.resolve() if make_absolute else resolved


def ensure_dir(path_spec: Union[str, Path]) -> Path:
    """
    Resolve a path and ensure the directory exists.
    
    Args:
        path_spec: Path string or Path object
        
    Returns:
        Path: Absolute, existing directory path
    """
    path = resolve_path(path_spec)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path_spec: Union[str, Path]) -> Path:
    """
    Resolve a path and ensure its parent directory exists.
    
    Args:
        path_spec: Path string or Path object (file path)
        
    Returns:
        Path: Absolute file path with existing parent directory
    """
    path = resolve_path(path_spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def config_to_absolute_paths(config: dict) -> dict:
    """
    Convert all relative paths in config dict to absolute paths.
    
    Modifies the config dict in-place for 'paths' section.
    
    Args:
        config: YAML config dict with 'paths' key
        
    Returns:
        dict: Same config with absolute paths
    """
    if "paths" in config:
        for key, path_str in config["paths"].items():
            config["paths"][key] = str(resolve_path(path_str))
    
    return config


if __name__ == "__main__":
    # Quick test
    print(f"Project root: {get_project_root()}")
    print(f"data/raw → {resolve_path('data/raw')}")
    print(f"configs/config.yaml → {resolve_path('configs/config.yaml')}")
