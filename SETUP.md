# Setup Guide for Llama Epsilon Level Sets Environment

This guide helps you set up the environment needed to run `find_llama_epsilon_level_sets.ipynb` on a remote machine with GPU support.

## Quick Start

### Complete Automated Setup (Recommended for New Deployments)

If you're setting up on a fresh remote machine with a mounted drive:

```bash
# 1. Copy this script to the remote machine
scp setup_remote_complete.sh user@remote-host:~/

# 2. SSH into the remote machine
ssh user@remote-host

# 3. Edit the mount path in the script (default: /mnt/data/llama-experiments)
nano setup_remote_complete.sh  # Change MOUNT_PATH variable

# 4. Run the complete setup
chmod +x setup_remote_complete.sh
./setup_remote_complete.sh
```

This single script will:
- Clone the repository to your mounted drive
- Validate all prerequisites
- Create and configure the Python environment
- Install all dependencies with proper CUDA support
- Handle HuggingFace authentication
- Pre-download the Llama-3.2-1B model

### Deploy Repository Only

If you just want to clone/update the repository to a mounted drive:

```bash
# Copy and run the deployment script
scp deploy_to_remote.sh user@remote-host:~/
ssh user@remote-host
nano deploy_to_remote.sh  # Edit MOUNT_PATH if needed
chmod +x deploy_to_remote.sh
./deploy_to_remote.sh
```

### Environment Setup Only (If Repository Already Exists)

If you already have the repository cloned, run one of these bootstrap scripts:

#### Option 1: Using Poetry (recommended)
```bash
chmod +x bootstrap.sh
./bootstrap.sh
```

### Option 2: Using pip only
```bash
chmod +x bootstrap_pip.sh
./bootstrap_pip.sh
```

Both scripts will:
- Check Python version (3.12+ required)
- Create a virtual environment
- Install all dependencies
- Handle HuggingFace authentication
- Pre-download the Llama-3.2-1B model
- Verify GPU availability

## Prerequisites

### On the Remote Machine (5090)

1. **NVIDIA GPU drivers** - The 5090 should have proper drivers installed
   ```bash
   nvidia-smi  # Should show your GPU
   ```

2. **CUDA toolkit** - CUDA 12.1 or compatible
   ```bash
   nvcc --version  # Check CUDA version
   ```

3. **Python 3.12+**
   ```bash
   python3 --version
   ```

4. **Git** (to clone the repository)
   ```bash
   git --version
   ```

### HuggingFace Setup (Required)

The Llama-3.2-1B model requires accepting Meta's license:

1. Go to https://huggingface.co/meta-llama/Llama-3.2-1B
2. Click "Agree and access repository"
3. Create an access token at https://huggingface.co/settings/tokens
4. When running the bootstrap script, log in with:
   ```bash
   huggingface-cli login
   ```
   Or set the environment variable:
   ```bash
   export HF_TOKEN="your_token_here"
   ```

## Manual Setup (Alternative)

If you prefer to set up manually:

```bash
# 1. Create virtual environment
python3 -m venv llama_epsilon_env
source llama_epsilon_env/bin/activate

# 2. Install PyTorch with CUDA
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies (choose one):
# Via Poetry:
pip install poetry
poetry install

# OR via pip:
pip install transformers accelerate huggingface-hub scikit-learn \
    numpy matplotlib notebook jupyterlab ipykernel ipywidgets \
    bertviz vec2text faiss-cpu cloudpickle umap-learn

# 4. Login to HuggingFace
huggingface-cli login

# 5. Pre-download model
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
"
```

## Running the Notebook

### On Remote Machine Directly

```bash
source llama_epsilon_env/bin/activate
jupyter lab
```

### Via SSH Port Forwarding (Recommended)

From your local machine:
```bash
ssh -L 8888:localhost:8888 user@remote-host
```

On the remote machine:
```bash
cd /path/to/input_space_gradients
source llama_epsilon_env/bin/activate
jupyter lab --no-browser --port=8888
```

Then open the URL shown in your local browser.

## Troubleshooting

### GPU Not Detected

If CUDA is not available in Python:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.version.cuda)         # Check CUDA version
```

If False:
- Verify nvidia-smi works
- Check PyTorch CUDA version matches your system
- Reinstall PyTorch with correct CUDA version

### Model Download Fails

If you get HTTP 401/403 errors:
1. Ensure you've accepted the license at https://huggingface.co/meta-llama/Llama-3.2-1B
2. Verify your token has read access
3. Re-login: `huggingface-cli login`

If you get rate limited (429 errors):
- The model will be cached after first successful download
- Set `LOCAL_ONLY = True` in `llama_models.py` after first download

### Out of Memory

The Llama-3.2-1B model requires ~5GB GPU memory:
```python
# Check available memory
import torch
print(torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
```

If running other processes on the GPU, consider:
- Using smaller batch sizes
- Enabling gradient checkpointing
- Clearing GPU memory: `torch.cuda.empty_cache()`

### Poetry Issues

If poetry install fails, use `bootstrap_pip.sh` instead.

## Verifying the Setup

After setup, verify everything works:

```bash
source llama_epsilon_env/bin/activate
python3 << 'EOF'
import torch
import transformers
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test model loading
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    local_files_only=True  # Should work if pre-downloaded
)
print("✓ Model loads successfully")
EOF
```

## File Structure

After setup, you should have:
```
input_space_gradients/
├── llama_epsilon_env/          # Virtual environment
├── find_llama_epsilon_level_sets.ipynb
├── llama_models.py
├── compute_batch_llama_epsilon_level_sets.py
├── pyproject.toml
├── poetry.lock
└── ~/.cache/huggingface/       # Model cache (default location)
```

## Additional Notes

- The model cache (~5GB) will be downloaded to `~/.cache/huggingface/hub/` by default
- You can change the cache location with `export HF_HOME=/custom/path`
- The 5090 has 32GB memory, which is more than sufficient for this model
- After first successful run, you can set `LOCAL_ONLY = True` in `llama_models.py` to avoid network calls
