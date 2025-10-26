#!/usr/bin/env python
"""
SAM2 Quick Test
Run after: mamba activate sam2
"""

import torch
import numpy as np
from sam2_wrapper import build_sam2_safe
from sam2.sam2_image_predictor import SAM2ImagePredictor

print("=" * 70)
print("SAM2 Verification - RTX Pro 6000 Blackwell")
print("=" * 70)

# Hardware
print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA: {torch.version.cuda}")

# Load model
print(f"\n⏳ Loading SAM2 tiny model...")
model = build_sam2_safe("tiny", device="cuda")
predictor = SAM2ImagePredictor(model)
print(f"✓ Model loaded successfully")

# Test inference
print(f"\n⏳ Running inference test (512x512 image)...")
dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
predictor.set_image(dummy_image)

masks, scores, _ = predictor.predict(
    point_coords=np.array([[256, 256]]),
    point_labels=np.array([1]),
    multimask_output=False
)

print(f"✓ Inference successful")
print(f"  - Output shape: {masks.shape}")
print(f"  - Confidence: {scores[0]:.3f}")
print(f"  - Segmented pixels: {masks[0].sum():,}")

print("\n" + "=" * 70)
print("✓ SAM2 is ready for production!")
print("=" * 70)

print("\n📝 Usage example:")
print("""
from sam2_wrapper import build_sam2_safe
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
from PIL import Image

# Load model once
model = build_sam2_safe("small", device="cuda")  # tiny, small, large
predictor = SAM2ImagePredictor(model)

# Load and process image
image = np.array(Image.open("photo.jpg"))
predictor.set_image(image)

# Point-based segmentation
masks, scores, _ = predictor.predict(
    point_coords=np.array([[x, y]]),
    point_labels=np.array([1]),
    multimask_output=False
)

# Access result
segmentation_mask = masks[0]  # Boolean array (H, W)
confidence = scores[0]  # Float 0-1
""")
