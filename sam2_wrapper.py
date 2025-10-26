#!/usr/bin/env python
"""
SAM2 Inference Wrapper - Properly fixes Hydra config path for editable installs

This wrapper solves the "Cannot find primary config 'tiny'" error by:
1. Finding the SAM2 configs directory
2. Properly initializing Hydra with that directory
3. Then calling build_sam2

Usage:
    from sam2_wrapper import build_sam2_safe
    model = build_sam2_safe("tiny", device="cuda")
"""

import os
import sys
from pathlib import Path
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

def build_sam2_safe(model_type="tiny", device="cuda"):
    """
    Build SAM2 model with proper Hydra config path handling
    
    Args:
        model_type: "tiny", "small", or "large"
        device: "cuda" or "cpu"
    
    Returns:
        SAM2 model ready for inference
    """
    
    # Find SAM2 configs directory
    try:
        import sam2
        sam2_path = Path(sam2.__file__).parent
        configs_dir = sam2_path / "configs"
        
        if not configs_dir.exists():
            raise FileNotFoundError(
                f"SAM2 configs not found at {configs_dir}\n"
                f"SAM2 is installed at: {sam2_path}\n"
                f"Make sure SAM2 is properly installed with: pip install -e ~/sam2"
            )
        
        configs_dir = str(configs_dir.absolute())
        
    except ImportError:
        raise ImportError(
            "SAM2 not installed. Run: pip install -e ~/sam2 in your mamba environment"
        )
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Initialize Hydra with the correct config directory
    try:
        with initialize_config_dir(version_base=None, config_dir=configs_dir):
            # Compose the config
            cfg = compose(config_name=f"{model_type}_hiera")
            
            # Now import and build SAM2
            from sam2.modeling.sam2 import SAM2
            
            # Build model from config
            model = SAM2(cfg)
            
            # Move to device
            model = model.to(device)
            model.eval()
            
            return model
            
    except Exception as e:
        raise RuntimeError(
            f"Failed to build SAM2 model '{model_type}':\n{str(e)}\n\n"
            f"Config directory: {configs_dir}\n"
            f"Available configs: {list((Path(configs_dir).glob('*.yaml')))}"
        )

__all__ = ['build_sam2_safe']
