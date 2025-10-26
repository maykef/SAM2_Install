#!/bin/bash
#
# SAM2 Installation Script for RTX Pro 6000 Blackwell
# Single command setup: bash setup.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

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
        
        # Check internet connectivity
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

# Initialize mamba for this script
eval "$("$MAMBA_ROOT/bin/mamba" shell hook --shell bash)"

# Verify mamba works
"$MAMBA_ROOT/bin/mamba" --version > /dev/null || error "Mamba failed to initialize"
log "Mamba initialized: $(mamba --version)"

# ============================================================================
# Step 2: Create sam2 mamba environment
# ============================================================================
log "Step 2/6: Creating sam2 environment from yml..."

# Get the directory where this script is located
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

# Create environment from yml
mamba env create -f "$ENV_YML" -y || error "Failed to create sam2 environment"
log "Environment created successfully"

# Activate environment
mamba activate sam2 || error "Failed to activate sam2 environment"
log "Environment activated"

# Upgrade PyTorch to nightly 2.10 with CUDA 12.8 for Blackwell sm_120
log "Upgrading PyTorch to nightly 2.10 for Blackwell optimization..."
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --quiet || \
    warn "PyTorch nightly upgrade failed, continuing with stable 2.4.1"

# Verify PyTorch CUDA support
python3 << 'PYTEST'
import torch
if not torch.cuda.is_available():
    print("WARNING: CUDA not available after PyTorch install")
else:
    print(f"✓ PyTorch {torch.__version__} with CUDA {torch.version.cuda}")
PYTEST

# ============================================================================
# Step 3: Setup model cache directories
# ============================================================================
log "Step 3/6: Setting up model cache directories..."

# Try /scratch first (faster for large models)
if [ -d "/scratch" ]; then
    if touch "/scratch/.write_test" 2>/dev/null; then
        rm "/scratch/.write_test"
        CACHE_DIR="$SCRATCH_CACHE"
        mkdir -p "$CACHE_DIR" || error "Failed to create $CACHE_DIR"
        log "Using /scratch for model cache: $CACHE_DIR"
    else
        warn "/scratch exists but not writable, trying with sudo..."
        sudo mkdir -p "$SCRATCH_CACHE" || true
        sudo chown -R "$USER:$USER" "$SCRATCH_CACHE" 2>/dev/null || true
        
        if touch "$SCRATCH_CACHE/.write_test" 2>/dev/null; then
            rm "$SCRATCH_CACHE/.write_test"
            CACHE_DIR="$SCRATCH_CACHE"
            log "Using /scratch for model cache (with permissions fix): $CACHE_DIR"
        else
            warn "/scratch not writable, falling back to home directory"
            CACHE_DIR="$HF_CACHE"
        fi
    fi
else
    warn "/scratch not available, using home directory for model cache"
    CACHE_DIR="$HF_CACHE"
fi

mkdir -p "$CACHE_DIR" || error "Failed to create cache directory: $CACHE_DIR"
log "Model cache directory ready: $CACHE_DIR"

# ============================================================================
# Step 4: Clone SAM2 repository
# ============================================================================
log "Step 4/6: Cloning SAM2 from GitHub..."

if [ -d "$SAM2_DIR" ]; then
    warn "SAM2 directory already exists at $SAM2_DIR, removing it..."
    rm -rf "$SAM2_DIR"
fi

# Check connectivity
if ! ping -c 1 github.com &> /dev/null; then
    error "No internet connection to GitHub"
fi

git clone https://github.com/facebookresearch/sam2.git "$SAM2_DIR" || \
    error "Failed to clone SAM2 repository"

log "SAM2 cloned to $SAM2_DIR"

# ============================================================================
# Step 5: Install SAM2 as editable package
# ============================================================================
log "Step 5/6: Installing SAM2 package..."

cd "$SAM2_DIR" || error "Failed to change to SAM2 directory"

# Ensure environment is still active
mamba activate sam2 || error "Failed to activate sam2 environment"

# Install in editable mode
pip install -e . --quiet || error "Failed to install SAM2 package"

log "SAM2 package installed (editable mode)"

# ============================================================================
# Step 6: Verification
# ============================================================================
log "Step 6/6: Verifying installation..."

# Test 1: Python can find sam2
python3 << 'PYTEST'
import sys
try:
    from sam2.build_sam import build_sam2
    print("✓ SAM2 import successful")
except ImportError as e:
    print(f"✗ Failed to import SAM2: {e}")
    sys.exit(1)
PYTEST

# Test 2: CUDA is available
python3 << 'PYTEST'
import sys
try:
    import torch
    if not torch.cuda.is_available():
        print("✗ CUDA not available - check PyTorch installation")
        sys.exit(1)
    print(f"✓ CUDA available: {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except Exception as e:
    print(f"✗ CUDA check failed: {e}")
    sys.exit(1)
PYTEST

# Test 3: HuggingFace Hub works
python3 << 'PYTEST'
import sys
try:
    from huggingface_hub import hf_hub_download
    print("✓ HuggingFace Hub available")
except ImportError as e:
    print(f"✗ HuggingFace Hub import failed: {e}")
    sys.exit(1)
PYTEST

# ============================================================================
# Success!
# ============================================================================
log "✓ Installation complete!"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "1. Close and reopen your terminal, or run: source ~/.bashrc"
echo "2. Activate environment: mamba activate sam2"
echo "3. Start using SAM2: python -c \"from sam2.build_sam import build_sam2\""
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  SAM2 location:    $SAM2_DIR"
echo "  Mamba location:   $MAMBA_ROOT"
echo "  Model cache:      $CACHE_DIR"
echo "  Environment:      sam2"
echo ""
echo "To test SAM2, run the example in the README"
exit 0
