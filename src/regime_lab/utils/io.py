"""I/O utilities for pickle operations and data serialization."""

import pickle
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def save_pickle(data: Any, filepath: str, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
    """Save data as pickle file.
    
    Args:
        data: Object to pickle
        filepath: Path where to save the pickle file
        protocol: Pickle protocol version (default: HIGHEST_PROTOCOL)
        
    Raises:
        OSError: If file cannot be written
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=protocol)


def load_pickle(filepath: str) -> Any:
    """Load data from pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Unpickled object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        pickle.PickleError: If unpickling fails
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data


def save_dataframe(df: pd.DataFrame, filepath: str, format: str = 'csv', **kwargs) -> None:
    """Save DataFrame to file.
    
    Args:
        df: DataFrame to save
        filepath: Path where to save the file
        format: File format ('csv', 'parquet', 'pickle')
        **kwargs: Additional arguments for the save method
        
    Raises:
        ValueError: If unsupported format is specified
        OSError: If file cannot be written
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'csv':
        df.to_csv(filepath, **kwargs)
    elif format.lower() == 'parquet':
        df.to_parquet(filepath, **kwargs)
    elif format.lower() == 'pickle':
        df.to_pickle(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_dataframe(filepath: str, format: Optional[str] = None) -> pd.DataFrame:
    """Load DataFrame from file.
    
    Args:
        filepath: Path to the file
        format: File format ('csv', 'parquet', 'pickle'). If None, infer from extension
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If unsupported format is specified
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    if format is None:
        format = filepath.suffix.lower().lstrip('.')
    
    if format == 'csv':
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
    elif format == 'parquet':
        return pd.read_parquet(filepath)
    elif format == 'pickle':
        return pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def file_exists(filepath: str) -> bool:
    """Check if file exists.
    
    Args:
        filepath: Path to check
        
    Returns:
        True if file exists, False otherwise
    """
    return Path(filepath).exists()


def get_file_size(filepath: str) -> int:
    """Get file size in bytes.
    
    Args:
        filepath: Path to the file
        
    Returns:
        File size in bytes
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    return filepath.stat().st_size
