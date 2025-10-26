#!/usr/bin/env python
"""
SAM2 wrapper - maps model type to correct config filename
"""

import os
from pathlib import Path

def build_sam2_safe(model_type="tiny", device="cuda"):
    """
    Build SAM2 model - maps model_type to config_file
    
    Args:
        model_type: "tiny", "small", "base", or "large"
        device: "cuda" or "cpu"
    
    Returns:
        SAM2 model ready for inference
    """
    
    # Map model types to actual config filenames (without .yaml)
    model_map = {
        "tiny": "sam2_hiera_t",
        "small": "sam2_hiera_s", 
        "base": "sam2_hiera_b",
        "large": "sam2_hiera_l",
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {', '.join(model_map.keys())}")
    
    config_file = model_map[model_type]
    
    # build_sam2 needs to be called from the sam2 directory
    import sam2
    sam2_dir = Path(sam2.__file__).parent.parent
    original_cwd = os.getcwd()
    
    try:
        os.chdir(sam2_dir)
        from sam2.build_sam import build_sam2
        model = build_sam2(config_file, device=device)
        return model
    finally:
        os.chdir(original_cwd)

__all__ = ['build_sam2_safe']
