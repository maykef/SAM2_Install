# SAM2 Setup for RTX Pro 6000 Blackwell

Professional-grade Segment Anything Model 2 installation for NVIDIA RTX Pro 6000 Blackwell GPU workstations (102GB VRAM). Single-command setup with comprehensive error handling.

## Quick Start

```bash
bash setup.sh
```

That's it. The script handles everything automatically:
- Installs Miniforge if needed
- Creates isolated `sam2` mamba environment
- Downloads SAM2 from official GitHub
- Installs PyTorch 2.10 nightly with CUDA 12.8
- Verifies GPU access
- Sets up model caching

**Installation time:** 10-15 minutes (mostly downloading PyTorch and models)

## System Requirements

### Hardware
- NVIDIA RTX Pro 6000 Blackwell (sm_120 compute capability)
- 102GB VRAM (or similar high-end RTX Pro GPU)
- AMD Threadripper 7970X or compatible CPU
- 50+ GB free disk space for PyTorch + SAM2 models
- 8TB NVMe scratch storage (recommended for model cache)
- Ubuntu 22.04 LTS or later (Desktop or Server)

### Software Prerequisites
- Git (`sudo apt install git`)
- curl (`sudo apt install curl`)
- wget (`sudo apt install wget`)
- GCC/G++ compiler (`sudo apt install build-essential`)
- Internet connection for initial setup

## What Gets Installed

### After running `setup.sh`, you get:

**Mamba environment:** `sam2`
- Python 3.11
- PyTorch 2.10.0 nightly with CUDA 12.8
- Transformers, HuggingFace Hub, timm
- OpenCV, scikit-image, scipy, numpy
- Jupyter (optional, for notebooks)

**SAM2 repository:** `~/sam2`
- Official SAM2 source code
- All model loading utilities
- Installed as editable package (pip install -e .)

**Model caching:** 
- Primary: `/scratch/models` (if writable)
- Fallback: `~/.cache/huggingface`
- Models auto-download on first use

**Total footprint:**
- Miniforge: ~300 MB
- mamba environment: ~3-4 GB
- SAM2 source + dependencies: ~2 GB
- Models (lazy-loaded): 100-500 MB each

## Detailed Installation Instructions

### Step 1: Prepare Your System

```bash
# Update package manager
sudo apt update && sudo apt upgrade -y

# Install build tools (if needed)
sudo apt install -y build-essential git curl wget

# Verify NVIDIA driver is installed
nvidia-smi

# You should see RTX Pro 6000 listed
```

### Step 2: Download Setup Files

Place these three files in a working directory:
- `setup.sh` - Main installer
- `sam2-env.yml` - Environment specification
- `README.md` - This documentation

```bash
# Make setup script executable
chmod +x setup.sh
```

### Step 3: Run Installation

```bash
cd /path/to/setup/files
bash setup.sh
```

The script will:
1. Check for Miniforge, install if missing
2. Create `sam2` mamba environment from `sam2-env.yml`
3. Clone SAM2 from GitHub to `~/sam2`
4. Install SAM2 as editable package
5. Verify PyTorch + CUDA + HuggingFace Hub
6. Print success message with next steps

### Step 4: Activate Environment

In a new terminal:

```bash
# Activate mamba environment
mamba activate sam2

# You should see (sam2) prefix in your prompt
```

Or source bashrc in current terminal:

```bash
source ~/.bashrc
mamba activate sam2
```

### Step 5: Verify Everything Works

```bash
python -c "from sam2.build_sam import build_sam2; print('✓ SAM2 ready')"
python -c "import torch; print(f'✓ CUDA: {torch.version.cuda}'); print(f'✓ GPU: {torch.cuda.get_device_name(0)}')"
```

## Verification: Testing Your Installation

### Quick Test (30 seconds)

```bash
mamba activate sam2
python << 'EOF'
import torch
from sam2.build_sam import build_sam2

# Verify components
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA: {torch.version.cuda}")
print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Build model (doesn't download until needed)
model = build_sam2("tiny", device="cuda")
print(f"✓ SAM2 tiny model loaded")
EOF
```

### Full Test (2-3 minutes, includes model download)

```bash
mamba activate sam2
python << 'EOF'
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Create dummy image (1024x1024 RGB)
image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

# Build model
device = "cuda" if torch.cuda.is_available() else "cpu"
sam2_model = build_sam2("tiny", device=device)
predictor = SAM2ImagePredictor(sam2_model)

# Run prediction
predictor.set_image(image)
masks, scores, logits = predictor.predict(
    point_coords=np.array([[512, 512]]),
    point_labels=np.array([1]),
    multimask_output=False
)

print(f"✓ Prediction successful")
print(f"✓ Mask shape: {masks.shape}")
print(f"✓ Confidence: {scores[0]:.3f}")
EOF
```

## Troubleshooting: Common Errors & Solutions

### Error 1: `bash: mamba: command not found`

**What you see:**
```
bash: mamba: command not found
```

**Why it happens:**
- Miniforge not installed
- Shell environment not reloaded
- MAMBA_ROOT_PREFIX pointing to wrong location

**How to fix:**
```bash
# Option 1: Reload shell (usually works)
source ~/.bashrc

# Option 2: Manually initialize mamba
eval "$($HOME/miniforge3/bin/mamba shell hook --shell bash)"
mamba --version

# Option 3: If still not found, reinstall
bash setup.sh
```

**Prevention:**
- Close and reopen terminal after setup
- setup.sh handles installation automatically

---

### Error 2: `CUDA not available in PyTorch`

**What you see:**
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Why it happens:**
- PyTorch installed without CUDA wheels
- Wrong CUDA version specified
- Driver-GPU mismatch (rare with RTX Pro 6000)

**How to fix:**
```bash
mamba activate sam2

# Reinstall PyTorch with correct CUDA 12.8
pip uninstall torch -y
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

**Prevention:**
- setup.sh uses pytorch channel with correct versions
- sam2-env.yml specifies pytorch-cuda=12.8
- No manual PyTorch installation needed

---

### Error 3: `cannot import name 'build_sam2' from 'sam2'`

**What you see:**
```
ImportError: cannot import name 'build_sam2' from 'sam2' (/home/user/sam2/__init__.py)
```

**Why it happens:**
- SAM2 package not properly installed
- Wrong Python path (using system Python instead of mamba)
- Corrupted SAM2 directory

**How to fix:**
```bash
# Ensure you're using mamba environment
which python  # Should show /home/user/miniforge3/envs/sam2/bin/python

# If not in mamba environment:
mamba activate sam2

# Reinstall SAM2
cd ~/sam2
pip install -e . --force-reinstall --no-deps

# Verify
python -c "from sam2.build_sam import build_sam2; print('OK')"
```

**Prevention:**
- setup.sh installs from official GitHub with pip install -e
- Always activate environment first: `mamba activate sam2`
- Script verifies import before declaring success

---

### Error 4: `Permission denied on /scratch`

**What you see:**
```
mkdir: cannot create directory '/scratch/models': Permission denied
```

**Why it happens:**
- /scratch owned by root or different user
- Insufficient permissions on /scratch directory
- /scratch doesn't exist and can't be created

**How to fix:**
```bash
# Check ownership
ls -ld /scratch

# Fix with sudo (setup.sh attempts this)
sudo mkdir -p /scratch/models /scratch/cache
sudo chown -R $USER:$USER /scratch/models /scratch/cache
sudo chmod -R 755 /scratch/models /scratch/cache

# Verify
touch /scratch/models/.test && rm /scratch/models/.test && echo "Writable"

# Rerun setup if needed
bash setup.sh
```

**Prevention:**
- setup.sh checks /scratch writability first
- Automatically falls back to `~/.cache/huggingface` if needed
- No failure if /scratch unavailable

---

### Error 5: `sam2.__file__ is None` (NoneType error)

**What you see:**
```
TypeError: expected str, bytes or os.PathLike object, not NoneType
  File "site-packages/sam2/...", line X, in <module>
    path = Path(sam2.__file__).parent
```

**Why it happens:**
- SAM2 package partially installed or corrupted
- Namespace package issue (rare)
- Mixing conda and pip installations

**How to fix:**
```bash
mamba activate sam2

# Clean uninstall
pip uninstall sam2 -y
rm -rf ~/sam2

# Reinstall
bash setup.sh
```

**Prevention:**
- setup.sh removes existing SAM2 before reinstalling
- Uses editable install (pip install -e .) consistently
- Verifies import after installation

---

### Error 6: `git clone fails - Could not resolve host`

**What you see:**
```
fatal: unable to access 'https://github.com/...': Could not resolve host
```

**Why it happens:**
- No internet connection
- GitHub unreachable
- DNS issues

**How to fix:**
```bash
# Check internet connectivity
ping github.com

# If GitHub is down, try again later
# If DNS broken:
cat /etc/resolv.conf  # Check if nameservers exist

# Alternatively, configure DNS
sudo bash -c 'echo "nameserver 8.8.8.8" > /etc/resolv.conf'
sudo bash -c 'echo "nameserver 8.8.4.4" >> /etc/resolv.conf'

# Retry setup
bash setup.sh
```

**Prevention:**
- setup.sh checks GitHub connectivity before cloning
- Fails gracefully with clear error message

---

### Error 7: `pip install -e . fails` with build errors

**What you see:**
```
error: command 'x86_64-linux-gnu-gcc' failed with exit status 1
```

or

```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Why it happens:**
- Missing compiler or build tools
- mamba environment missing cmake/ninja
- Python development headers missing

**How to fix:**
```bash
mamba activate sam2

# Install build dependencies
mamba install -y cmake ninja

# Ensure you have system build tools
sudo apt install -y build-essential python3-dev

# Try installation again
cd ~/sam2
pip install -e . --verbose
```

**Prevention:**
- sam2-env.yml includes cmake, ninja, cxx-compiler
- setup.sh uses mamba environment with all dependencies
- No need for manual tool installation

---

### Error 8: `No space left on device`

**What you see:**
```
OSError: [Errno 28] No space left on device
```

**Why it happens:**
- Less than 50 GB free disk space
- Model cache filling up quickly
- System partition too small

**How to fix:**
```bash
# Check disk space
df -h

# Clear old models if needed
rm -rf ~/.cache/huggingface/hub/*

# If using /scratch, check there too
df -h /scratch

# Free up space, then retry setup
bash setup.sh
```

**Prevention:**
- setup.sh checks space implicitly (will fail to download if insufficient)
- Use separate /scratch partition on 8TB NVMe
- Models are lazy-loaded (only download when used)

---

### Error 9: `AssertionError: CUDA SM capability >= 7.0 required`

**What you see:**
```
AssertionError: CUDA SM capability of your device is not supported
```

**Why it happens:**
- Wrong GPU or old GPU without Tensor cores
- CUDA capability mismatch (extremely rare with RTX Pro 6000)

**How to fix:**
```bash
# Verify GPU
nvidia-smi

# Check compute capability
python -c "import torch; print(torch.cuda.get_device_capability(0))"

# RTX Pro 6000 should report sm_120
# If different, you have the wrong GPU
```

**Prevention:**
- This is a hardware issue, not setup issue
- RTX Pro 6000 Blackwell is fully supported

---

### Error 10: `ImportError: libcuda.so.1 not found`

**What you see:**
```
ImportError: libcuda.so.1: cannot open shared object file: No such file or directory
```

**Why it happens:**
- NVIDIA driver not installed
- CUDA toolkit not installed
- LD_LIBRARY_PATH misconfigured

**How to fix:**
```bash
# Install NVIDIA driver
sudo apt install nvidia-driver-550

# Verify driver is installed
nvidia-smi

# If still broken, install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Reboot if driver was installed
sudo reboot
```

**Prevention:**
- Ensure `nvidia-smi` works BEFORE running setup.sh
- Driver installation is system-level, not setup responsibility

---

## Usage Examples

### Example 1: Segment Objects from Point Clicks

```python
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load image
image = np.array(Image.open("photo.jpg"))

# Initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
sam2_model = build_sam2("large", device=device)  # tiny, small, large
predictor = SAM2ImagePredictor(sam2_model)

# Set image
predictor.set_image(image)

# Click on object (point_coords format: [[x, y]])
point_coords = np.array([[500, 400]])  # Single click
point_labels = np.array([1])  # 1 = foreground, 0 = background

# Get segmentation
masks, scores, logits = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    multimask_output=False
)

# masks shape: (1, H, W)
print(f"Segmented area: {masks[0].sum()} pixels")

# Visualize
segmented = image.copy()
segmented[masks[0] > 0] = [255, 0, 0]  # Red overlay
Image.fromarray(segmented).save("result.jpg")
```

### Example 2: Batch Processing Images

```python
import torch
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
from PIL import Image

# Setup
device = "cuda"
model = build_sam2("tiny", device=device)
predictor = SAM2ImagePredictor(model)

# Process directory
image_dir = Path("images/")
for image_path in sorted(image_dir.glob("*.jpg")):
    image = np.array(Image.open(image_path))
    predictor.set_image(image)
    
    # Your segmentation logic here
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[512, 512]]),
        point_labels=np.array([1]),
    )
    
    print(f"Processed {image_path.name}: mask confidence {scores[0]:.3f}")
```

### Example 3: Box-Based Segmentation

```python
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

device = "cuda"
model = build_sam2("large", device=device)
predictor = SAM2ImagePredictor(model)

# Set image
image = np.array(Image.open("photo.jpg"))
predictor.set_image(image)

# Define bounding box [x1, y1, x2, y2]
box = np.array([100, 100, 400, 400])

# Segment from box
masks, scores, _ = predictor.predict(
    box=box,
    multimask_output=False
)

print(f"Box segmentation confidence: {scores[0]:.3f}")
```

### Example 4: Multiple Points (Refinement)

```python
import torch
import numpy as np
from PIL import Image

# Initialize predictor
device = "cuda"
model = build_sam2("large", device=device)
predictor = SAM2ImagePredictor(model)

image = np.array(Image.open("photo.jpg"))
predictor.set_image(image)

# Foreground points (multiple)
fg_points = np.array([
    [200, 200],
    [300, 250],
    [250, 300]
])

# Background points
bg_points = np.array([
    [100, 100]
])

# Combine with labels
all_points = np.vstack([fg_points, bg_points])
all_labels = np.array([1, 1, 1, 0])  # 1=foreground, 0=background

# Segment
masks, scores, _ = predictor.predict(
    point_coords=all_points,
    point_labels=all_labels,
    multimask_output=False
)
```

## Directory Structure

After running `setup.sh`, your system will have:

```
$HOME/
├── miniforge3/                 # Miniforge installation
│   ├── bin/mamba
│   ├── envs/
│   │   └── sam2/              # Mamba environment
│   │       ├── bin/python
│   │       ├── lib/python3.11/site-packages/
│   │       │   ├── torch/
│   │       │   ├── sam2/      # SAM2 package (symlink)
│   │       │   └── ...
│   │       └── ...
│   └── ...
│
├── sam2/                       # SAM2 source code (editable install)
│   ├── sam2/
│   │   ├── __init__.py
│   │   ├── build_sam.py
│   │   ├── sam2_image_predictor.py
│   │   └── ...
│   ├── setup.py
│   ├── README.md
│   └── ...
│
├── .cache/huggingface/         # HuggingFace model cache
│   └── hub/
│       └── models--...
│
└── .bashrc                     # Updated with mamba init (if needed)

/scratch/                       # Fast model cache (if available)
├── models/
│   └── ...
└── cache/
    └── ...
```

## Common Errors Section: Quick Reference

| Error | Cause | Solution |
|-------|-------|----------|
| `mamba: command not found` | Shell not reloaded after install | `source ~/.bashrc` or reopen terminal |
| `CUDA not available` | PyTorch missing CUDA wheels | Reinstall PyTorch from pytorch nightly index |
| `cannot import build_sam2` | Wrong Python or corrupted install | `mamba activate sam2` then reinstall SAM2 |
| `Permission denied /scratch` | /scratch not writable | `sudo chown -R $USER:$USER /scratch` |
| `sam2.__file__ is None` | Corrupted package | `pip uninstall sam2 -y && bash setup.sh` |
| `git clone fails` | No internet or GitHub down | Check connectivity with `ping github.com` |
| `pip install -e . fails` | Missing build tools | `mamba install -y cmake ninja` |
| `No space left on device` | Insufficient disk space | Free space or use separate /scratch partition |
| `CUDA SM capability >= 7.0` | Wrong GPU (impossible with RTX Pro 6000) | Verify GPU with `nvidia-smi` |
| `libcuda.so.1 not found` | NVIDIA driver not installed | `sudo apt install nvidia-driver-550` |

## Model Variants

SAM2 is available in three sizes. Choose based on accuracy vs. speed tradeoff:

| Model | Parameters | Memory | Speed | Accuracy | Use Case |
|-------|-----------|--------|-------|----------|----------|
| **tiny** | 38M | 4-6 GB | Fast | Good | Real-time, single objects |
| **small** | 47M | 6-8 GB | Balanced | Very Good | General purpose |
| **large** | 81M | 12-16 GB | Slower | Excellent | Batch processing, complex scenes |

```python
# Load different sizes
model_tiny = build_sam2("tiny", device="cuda")     # Fast
model_small = build_sam2("small", device="cuda")   # Balanced
model_large = build_sam2("large", device="cuda")   # High accuracy
```

With 102GB VRAM on RTX Pro 6000, you can even run multiple models in parallel.

## Environment Variables

If you need to customize behavior, set these before activating:

```bash
# Model cache location (default: ~/.cache/huggingface)
export HF_HOME=/scratch/cache
export HF_HUB_CACHE=/scratch/cache/hub_cache

# PyTorch settings
export TORCH_HOME=/scratch/torch_cache
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU

# Then activate
mamba activate sam2
```

## Performance Notes

- **Cold start:** First model load downloads ~400-600 MB from HuggingFace Hub
- **Inference speed:** ~50-200 ms per image (tiny to large) with batching
- **Memory:** Large model uses 10-16 GB with full resolution images
- **Batch processing:** 102GB VRAM allows processing high-resolution images without memory swaps

## Support & Resources

- **SAM2 GitHub:** https://github.com/facebookresearch/sam2
- **Official Docs:** https://github.com/facebookresearch/sam2/blob/main/README.md
- **Model Hub:** https://huggingface.co/models?search=sam2
- **NVIDIA Blackwell:** https://nvidia.com/en-us/data-center/blackwell/

## FAQ

**Q: Can I use the system Python instead of mamba?**
A: Not recommended. Mamba isolates dependencies and ensures consistency. The setup uses mamba by design.

**Q: Does setup.sh need internet every time?**
A: Only the first time. After installation, models are cached locally.

**Q: Can I run multiple SAM2 models simultaneously?**
A: Yes, but memory-dependent. Large model alone uses 12-16 GB, leaving 86-90 GB for others.

**Q: How do I update SAM2?**
A: `cd ~/sam2 && git pull` (already installed as editable package)

**Q: Why PyTorch nightly instead of stable?**
A: Nightly build (2.10) has optimizations for Blackwell. Stable versions predate sm_120 support.

---

**Last Updated:** October 2025
**Compatible With:** SAM2 (Meta), PyTorch 2.10+, CUDA 12.8, RTX Pro 6000 Blackwell
