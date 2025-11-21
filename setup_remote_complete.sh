#!/bin/bash
set -e

echo "=================================================="
echo "Complete Remote Setup for Llama Epsilon Level Sets"
echo "=================================================="
echo ""

# Configuration - EDIT THESE AS NEEDED
REPO_URL="git@github.com:kyle-pena-nlp/token-input-embedding-gradients.git"
MOUNT_PATH="/mnt/data/llama-experiments"  # Change this to your mounted drive path
REPO_NAME="input_space_gradients"
ENV_NAME="llama_epsilon_env"

# Allow override via environment variables
MOUNT_PATH="${DEPLOY_MOUNT_PATH:-$MOUNT_PATH}"

echo "Configuration:"
echo "  Repository: $REPO_URL"
echo "  Mount path: $MOUNT_PATH"
echo "  Target directory: $MOUNT_PATH/$REPO_NAME"
echo "  Python environment: $ENV_NAME"
echo ""

# ============================================
# STEP 1: Validate Prerequisites
# ============================================
echo "=================================================="
echo "Step 1: Validating Prerequisites"
echo "=================================================="

# Check mount point
if [ ! -d "$MOUNT_PATH" ]; then
    echo "❌ ERROR: Mount path $MOUNT_PATH does not exist"
    read -p "Create it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo mkdir -p "$MOUNT_PATH"
        sudo chown $USER:$USER "$MOUNT_PATH"
    else
        exit 1
    fi
fi

if [ ! -w "$MOUNT_PATH" ]; then
    echo "❌ ERROR: Cannot write to $MOUNT_PATH"
    exit 1
fi
echo "✓ Mount path accessible"

# Check git
if ! command -v git &> /dev/null; then
    echo "❌ ERROR: git is not installed"
    echo "Install with: sudo apt install git"
    exit 1
fi
echo "✓ git installed"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: python3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $PYTHON_VERSION"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)"; then
    echo "❌ ERROR: Python 3.12+ required"
    exit 1
fi

# Check NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo "⚠ WARNING: nvidia-smi not found"
fi

# ============================================
# STEP 2: Clone/Update Repository
# ============================================
echo ""
echo "=================================================="
echo "Step 2: Repository Setup"
echo "=================================================="

TARGET_PATH="$MOUNT_PATH/$REPO_NAME"

if [ -d "$TARGET_PATH" ]; then
    echo "Repository already exists at $TARGET_PATH"
    read -p "Update it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$TARGET_PATH"
        echo "Pulling latest changes..."
        git pull origin main || git pull origin master
        echo "✓ Repository updated"
    fi
else
    echo "Cloning repository..."
    cd "$MOUNT_PATH"
    git clone "$REPO_URL" "$REPO_NAME"
    echo "✓ Repository cloned"
fi

cd "$TARGET_PATH"

# ============================================
# STEP 3: Python Environment Setup
# ============================================
echo ""
echo "=================================================="
echo "Step 3: Python Environment"
echo "=================================================="

if [ -d "$ENV_NAME" ]; then
    echo "Virtual environment already exists"
    read -p "Recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$ENV_NAME"
        python3 -m venv "$ENV_NAME"
        echo "✓ Environment recreated"
    fi
else
    echo "Creating virtual environment..."
    python3 -m venv "$ENV_NAME"
    echo "✓ Environment created"
fi

# Activate environment
source "$ENV_NAME/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools > /dev/null

# ============================================
# STEP 4: Install Dependencies
# ============================================
echo ""
echo "=================================================="
echo "Step 4: Installing Dependencies"
echo "=================================================="

# Detect CUDA version
CUDA_VERSION=""
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "Detected CUDA version: $CUDA_VERSION"
fi

# Install PyTorch
if [ -n "$CUDA_VERSION" ]; then
    echo "Installing PyTorch with CUDA support..."
    if [[ "$CUDA_VERSION" == "12.1" ]] || [[ "$CUDA_VERSION" == "12."* ]]; then
        pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu121
    else
        pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
    fi
else
    echo "Installing PyTorch (CPU fallback)..."
    pip install torch==2.7.0
fi

# Install all other dependencies
echo "Installing other dependencies..."
pip install \
    transformers \
    accelerate \
    huggingface-hub \
    scikit-learn \
    numpy \
    matplotlib==3.10.3 \
    notebook==7.4.3 \
    jupyterlab \
    ipykernel \
    ipywidgets \
    bertviz \
    vec2text \
    faiss-cpu \
    cloudpickle \
    umap-learn \
    fastapi==0.115.12 \
    uvicorn==0.34.3

echo "✓ Dependencies installed"

# ============================================
# STEP 5: HuggingFace Authentication
# ============================================
echo ""
echo "=================================================="
echo "Step 5: HuggingFace Authentication"
echo "=================================================="

if [ -n "$HF_TOKEN" ]; then
    echo "Using HF_TOKEN from environment"
    huggingface-cli login --token "$HF_TOKEN"
elif huggingface-cli whoami &> /dev/null; then
    echo "✓ Already authenticated"
else
    echo "HuggingFace authentication required"
    echo ""
    echo "Steps:"
    echo "1. Visit: https://huggingface.co/meta-llama/Llama-3.2-1B"
    echo "2. Accept the license agreement"
    echo "3. Create token: https://huggingface.co/settings/tokens"
    echo ""
    read -p "Press Enter when ready to login..."
    huggingface-cli login
fi

# ============================================
# STEP 6: Pre-download Model
# ============================================
echo ""
echo "=================================================="
echo "Step 6: Pre-downloading Llama-3.2-1B"
echo "=================================================="

python3 << 'EOF'
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "meta-llama/Llama-3.2-1B"

try:
    print(f"Downloading {MODEL}...")
    print("(This may take several minutes)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    print("✓ Tokenizer downloaded")

    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True)
    print("✓ Model downloaded")

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"\nModel: {param_count:.2f}B parameters")

    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Memory: {mem_gb:.1f} GB")
    else:
        print("⚠ CUDA not available - will run on CPU")

except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "Model download failed - see error above"
    exit 1
fi

# ============================================
# COMPLETION
# ============================================
echo ""
echo "=================================================="
echo "Setup Complete! 🎉"
echo "=================================================="
echo ""
echo "Project location: $TARGET_PATH"
echo "Virtual environment: $ENV_NAME"
echo ""
echo "To start working:"
echo "  cd $TARGET_PATH"
echo "  source $ENV_NAME/bin/activate"
echo "  jupyter lab --no-browser"
echo ""
echo "For SSH access from your local machine:"
echo "  ssh -L 8888:localhost:8888 user@$(hostname)"
echo ""
echo "Then open the Jupyter URL in your local browser"
echo ""
