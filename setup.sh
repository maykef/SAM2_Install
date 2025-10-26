#!/bin/bash
#
# SAM2 Installation for RTX Pro 6000 Blackwell
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log() { echo -e "${GREEN}[SAM2]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

SAM2_DIR="$HOME/sam2"
MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-$HOME/miniforge3}"

# Step 1: Miniforge
log "Step 1/6: Checking Miniforge..."
if [ ! -d "$MAMBA_ROOT" ]; then
    log "Installing Miniforge..."
    curl -fsSL https://github.com/conda-forge/miniforge/releases/download/24.11.2-0/Miniforge3-Linux-x86_64.sh -o /tmp/mf.sh
    bash /tmp/mf.sh -b -p "$MAMBA_ROOT"
    rm /tmp/mf.sh
fi
eval "$("$MAMBA_ROOT/bin/mamba" shell hook --shell bash)"
log "Mamba OK: $(mamba --version)"

# Step 2: Environment
log "Step 2/6: Creating mamba environment..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_YML="$SCRIPT_DIR/sam2-env.yml"
[ -f "$ENV_YML" ] || error "sam2-env.yml not found in $SCRIPT_DIR"
mamba env list | grep -q "^sam2 " && mamba remove -n sam2 -y --all
mamba env create -f "$ENV_YML" -y
mamba activate sam2
log "Environment created"

# Step 3: PyTorch nightly
log "Step 3/6: Installing PyTorch 2.10 nightly with CUDA 12.8..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --quiet
python3 -c "import torch; assert torch.cuda.is_available(); print(f'✓ PyTorch {torch.__version__} + CUDA {torch.version.cuda}')"

# Step 4: Dependencies
log "Step 4/6: Installing dependencies..."
pip install --quiet transformers huggingface_hub safetensors timm tqdm pyyaml omegaconf
log "Dependencies OK"

# Step 5: SAM2
log "Step 5/6: Installing SAM2..."
[ -d "$SAM2_DIR" ] && rm -rf "$SAM2_DIR"
git clone https://github.com/facebookresearch/sam2.git "$SAM2_DIR"
cd "$SAM2_DIR"
mamba activate sam2
pip install -e . --quiet
log "SAM2 installed at $SAM2_DIR"

# Download config files (required for build_sam2)
log "Downloading SAM2 config files..."
mkdir -p "$SAM2_DIR/sam2/configs"
cd "$SAM2_DIR/sam2/configs"
curl -fsSL -O https://huggingface.co/facebook/sam2-hiera-tiny/resolve/main/sam2_hiera_t.yaml
curl -fsSL -O https://huggingface.co/facebook/sam2-hiera-small/resolve/main/sam2_hiera_s.yaml
curl -fsSL -O https://huggingface.co/facebook/sam2-hiera-base/resolve/main/sam2_hiera_b.yaml
curl -fsSL -O https://huggingface.co/facebook/sam2-hiera-large/resolve/main/sam2_hiera_l.yaml
log "Configs downloaded"

# Verify
log "Step 6/6: Verifying..."
cd "$SAM2_DIR"
python3 -c "from sam2.build_sam import build_sam2; print('✓ Import OK')"

log "✓ Installation complete!"
echo ""
echo "USAGE:"
echo "  cd ~/sam2"
echo "  mamba activate sam2"
echo "  python your_script.py"
echo ""
echo "In Python, run from ~/sam2 directory and use:"
echo "  from sam2.build_sam import build_sam2"
echo "  model = build_sam2('tiny', device='cuda')"
echo ""
