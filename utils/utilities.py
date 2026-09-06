"""
Utility functions for the invoice extraction pipeline.
"""

import os
import torch
import platform


def create_directories(directories):
    """
    Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directory ready: {directory}/")


def get_optimal_device(device="auto"):
    """
    Determine the best available device for PyTorch.
    
    Args:
        device: Requested device ('auto', 'cuda', 'mps', 'cpu')
        
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if device != "auto":
        # User specified a device
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif device == "cpu":
            return "cpu"
        else:
            print(f"⚠️  Requested device '{device}' not available, falling back to auto-detection")
    
    # Auto-detection
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        # Check if MPS is built and available
        try:
            # Test MPS availability
            torch.zeros(1).to('mps')
            return "mps"
        except:
            print("⚠️  MPS detected but not functional, using CPU")
            return "cpu"
    else:
        return "cpu"


def get_device_info():
    """
    Get detailed information about available compute devices.
    
    Returns:
        Dictionary containing device information
    """
    info = {
        "Platform": platform.system(),
        "Python Version": platform.python_version(),
        "PyTorch Version": torch.__version__,
    }
    
    # CUDA Info
    if torch.cuda.is_available():
        info["CUDA Available"] = "Yes"
        info["CUDA Version"] = torch.version.cuda
        info["GPU Count"] = torch.cuda.device_count()
        info["GPU Name"] = torch.cuda.get_device_name(0)
        info["Selected Device"] = "CUDA (GPU)"
    # MPS Info (Apple Silicon)
    elif torch.backends.mps.is_available():
        try:
            torch.zeros(1).to('mps')
            info["MPS Available"] = "Yes (Apple Silicon GPU)"
            info["Selected Device"] = "MPS (Apple Silicon GPU)"
        except:
            info["MPS Available"] = "Detected but not functional"
            info["Selected Device"] = "CPU (Fallback)"
    else:
        info["CUDA Available"] = "No"
        info["MPS Available"] = "No"
        info["Selected Device"] = "CPU"
    
    return info


def format_bbox(bbox):
    """
    Format bounding box coordinates for display.
    
    Args:
        bbox: List of coordinates [x1, y1, x2, y2]
        
    Returns:
        Formatted string
    """
    if not bbox:
        return "None"
    return f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"