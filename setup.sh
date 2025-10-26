#!/bin/bash
#
# SAM2 Installation Script for RTX Pro 6000 Blackwell
# Minimal, working approach: base environment + PyTorch via pip
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
SAM2_DIR="$HOME/sam2"
MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-$HOME/miniforge3}"
HF_CACHE="${HOME}/.cache/huggingface"
SCRATCH_CACHE="/scratch/models"

# Helper functions
log() {
    echo -e "${GREEN}[SAM2]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# ============================================================================
# Step 1: Check and install Miniforge if needed
# ============================================================================
log "Step 1/6: Checking Miniforge installation..."

if [ ! -d "$MAMBA_ROOT" ]; then
    if [ -d "$HOME/mambaforge" ]; then
        MAMBA_ROOT="$HOME/mambaforge"
        log "Found Mambaforge at $MAMBA_ROOT"
    elif [ -d "$HOME/anaconda3" ]; then
        MAMBA_ROOT="$HOME/anaconda3"
        log "Found Anaconda at $MAMBA_ROOT"
    else
        log "Miniforge not found. Installing..."
        
        if ! ping -c 1 github.com &> /dev/null; then
            error "No internet connection to GitHub"
        fi
        
        MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/24.11.2-0/Miniforge3-Linux-x86_64.sh"
        MINIFORGE_INSTALLER="/tmp/miniforge.sh"
        
        curl -fsSL "$MINIFORGE_URL" -o "$MINIFORGE_INSTALLER" || error "Failed to download Miniforge"
        bash "$MINIFORGE_INSTALLER" -b -p "$MAMBA_ROOT" || error "Failed to install Miniforge"
        rm -f "$MINIFORGE_INSTALLER"
        log "Miniforge installed to $MAMBA_ROOT"
    fi
fi

eval "$("$MAMBA_ROOT/bin/mamba" shell hook --shell bash)"
"$MAMBA_ROOT/bin/mamba" --version > /dev/null || error "Mamba failed to initialize"
log "Mamba initialized: $(mamba --version)"

# ============================================================================
# Step 2: Create sam2 mamba environment (minimal, no PyTorch yet)
# ============================================================================
log "Step 2/6: Creating sam2 environment..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_YML="$SCRIPT_DIR/sam2-env.yml"

if [ ! -f "$ENV_YML" ]; then
    error "sam2-env.yml not found at $ENV_YML"
fi

# Remove existing environment if present
if mamba env list | grep -q "^sam2 "; then
    warn "sam2 environment already exists. Removing it..."
    mamba remove -n sam2 -y --all || true
fi

# Create environment from yml (minimal, no PyTorch)
mamba env create -f "$ENV_YML" -y || error "Failed to create sam2 environment"
log "Base environment created"

# Activate environment
mamba activate sam2 || error "Failed to activate sam2 environment"
log "Environment activated"

# ============================================================================
# Step 3: Install PyTorch nightly with CUDA 12.8 for Blackwell
# ============================================================================
log "Step 3/6: Installing PyTorch 2.10 nightly with CUDA 12.8..."

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --quiet || \
    error "Failed to install PyTorch nightly"

log "PyTorch installed"

# Verify CUDA
python3 << 'PYTEST'
import torch
print(f"✓ PyTorch {torch.__version__}")
print(f"✓ CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    exit(1)
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PYTEST

# ============================================================================
# Step 4: Install HuggingFace ecosystem via pip
# ============================================================================
log "Step 4/6: Installing HuggingFace and SAM2 dependencies..."

pip install --quiet \
    'transformers>=4.36' \
    'huggingface_hub>=0.19' \
    'safetensors>=0.4' \
    'timm>=0.9' \
    'tqdm>=4.66' \
    'pyyaml>=6.0' \
    'omegaconf>=2.3' || error "Failed to install dependencies"

log "Dependencies installed"

# ============================================================================
# Step 5: Setup model cache directories
# ============================================================================
log "Step 5/6: Setting up model cache..."

if [ -d "/scratch" ]; then
    if touch "/scratch/.write_test" 2>/dev/null; then
        rm "/scratch/.write_test"
        mkdir -p "$SCRATCH_CACHE"
        CACHE_DIR="$SCRATCH_CACHE"
        log "Model cache: $SCRATCH_CACHE (fast NVMe)"
    else
        warn "/scratch not writable, using home directory"
        CACHE_DIR="$HF_CACHE"
    fi
else
    CACHE_DIR="$HF_CACHE"
fi

mkdir -p "$CACHE_DIR"
log "Cache directory: $CACHE_DIR"

# ============================================================================
# Step 6: Clone and install SAM2
# ============================================================================
log "Step 6/6: Installing SAM2..."

if [ -d "$SAM2_DIR" ]; then
    warn "SAM2 directory exists, removing it..."
    rm -rf "$SAM2_DIR"
fi

if ! ping -c 1 github.com &> /dev/null; then
    error "No internet connection to GitHub"
fi

git clone https://github.com/facebookresearch/sam2.git "$SAM2_DIR" || \
    error "Failed to clone SAM2"

cd "$SAM2_DIR"
mamba activate sam2  # Ensure environment still active
pip install -e . --quiet || error "Failed to install SAM2"

log "SAM2 installed"

# ============================================================================
# Verification
# ============================================================================
log "Verifying installation..."

python3 << 'PYTEST'
import sys
try:
    from sam2.build_sam import build_sam2
    print("✓ SAM2 import OK")
    
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.version.cuda}")
    else:
        print("✗ CUDA not available")
        sys.exit(1)
    
    from huggingface_hub import hf_hub_download
    print("✓ HuggingFace Hub OK")
    
except Exception as e:
    print(f"✗ Verification failed: {e}")
    sys.exit(1)
PYTEST

# ============================================================================
# Success
# ============================================================================
log "✓ Installation complete!"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "1. Close and reopen terminal, or: source ~/.bashrc"
echo "2. Activate: mamba activate sam2"
echo "3. Test: python -c \"from sam2.build_sam import build_sam2; print('Ready')\""
echo ""
echo -e "${GREEN}Locations:${NC}"
echo "  SAM2:        $SAM2_DIR"
echo "  Mamba:       $MAMBA_ROOT"
echo "  Cache:       $CACHE_DIR"
echo ""
exit 0
