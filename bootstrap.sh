#!/bin/bash
set -e

echo "=================================================="
echo "Bootstrap Script for Llama Epsilon Level Sets"
echo "=================================================="

# Check if we're on the target machine
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    echo "⚠ WARNING: No NVIDIA GPU detected (nvidia-smi not found)"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.12"
echo "Python version: $PYTHON_VERSION"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)"; then
    echo "❌ ERROR: Python 3.12+ required, found $PYTHON_VERSION"
    exit 1
fi
echo "✓ Python version OK"

# Create virtual environment
ENV_NAME="llama_epsilon_env"
if [ -d "$ENV_NAME" ]; then
    echo "⚠ Virtual environment '$ENV_NAME' already exists"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$ENV_NAME"
    else
        echo "Using existing environment"
    fi
fi

if [ ! -d "$ENV_NAME" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$ENV_NAME"
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
source "$ENV_NAME/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install poetry
echo "Installing poetry..."
pip install poetry

# Install dependencies from pyproject.toml
echo "Installing project dependencies..."
poetry install

# Install additional dependencies that might be implicit
echo "Installing transformers, scikit-learn, and accelerate..."
pip install transformers scikit-learn accelerate huggingface-hub

# Check for HuggingFace token
echo ""
echo "=================================================="
echo "HuggingFace Authentication"
echo "=================================================="
if [ -n "$HF_TOKEN" ]; then
    echo "✓ HF_TOKEN environment variable found"
    huggingface-cli login --token "$HF_TOKEN"
elif huggingface-cli whoami &> /dev/null; then
    echo "✓ Already logged in to HuggingFace"
else
    echo "⚠ HuggingFace authentication required for Llama models"
    echo ""
    echo "Please:"
    echo "1. Go to https://huggingface.co/meta-llama/Llama-3.2-1B"
    echo "2. Accept the license agreement"
    echo "3. Create a token at https://huggingface.co/settings/tokens"
    echo "4. Run: huggingface-cli login"
    echo ""
    read -p "Press Enter when ready to continue with login..."
    huggingface-cli login
fi

# Pre-download the model
echo ""
echo "=================================================="
echo "Pre-downloading Llama-3.2-1B Model"
echo "=================================================="
python3 << 'EOF'
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "meta-llama/Llama-3.2-1B"

print(f"Downloading tokenizer for {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
print("✓ Tokenizer downloaded")

print(f"Downloading model for {MODEL}...")
print("(This may take several minutes depending on your connection)")
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True)
print("✓ Model downloaded")

print(f"\nModel size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
print(f"Cache location: {tokenizer.pretrained_init_configuration.get('_name_or_path', 'default HF cache')}")

# Verify GPU availability
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠ WARNING: CUDA not available")
EOF

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "To use the environment:"
echo "  1. Activate: source $ENV_NAME/bin/activate"
echo "  2. Start Jupyter: jupyter lab"
echo "  3. Open: find_llama_epsilon_level_sets.ipynb"
echo ""
echo "To run on remote machine:"
echo "  1. SSH with port forwarding: ssh -L 8888:localhost:8888 user@host"
echo "  2. On remote: source $ENV_NAME/bin/activate && jupyter lab --no-browser"
echo "  3. Open the URL shown in your local browser"
echo ""
