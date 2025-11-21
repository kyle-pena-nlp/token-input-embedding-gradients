# Deployment Scripts Reference

Quick reference for deploying to your remote machine with the 5090 GPU.

## Available Scripts

### 1. `setup_remote_complete.sh` - Full Automated Setup ⭐
**Best for**: Fresh deployment to a new machine

Handles everything in one go:
- Clones repository from GitHub
- Sets up Python environment
- Installs all dependencies
- Configures HuggingFace
- Downloads the model

**Usage**:
```bash
# On your local machine
scp setup_remote_complete.sh user@remote-host:~/

# On remote machine
ssh user@remote-host
nano setup_remote_complete.sh  # Edit MOUNT_PATH (default: /mnt/data/llama-experiments)
chmod +x setup_remote_complete.sh
./setup_remote_complete.sh
```

### 2. `deploy_to_remote.sh` - Repository Only
**Best for**: Just cloning/updating the code

Only handles git operations:
- Clones repository if not exists
- Updates repository if already exists
- Validates required files

**Usage**:
```bash
# On your local machine
scp deploy_to_remote.sh user@remote-host:~/

# On remote machine
ssh user@remote-host
nano deploy_to_remote.sh  # Edit MOUNT_PATH if needed
chmod +x deploy_to_remote.sh
./deploy_to_remote.sh

# Then run bootstrap separately
cd /mnt/data/llama-experiments/input_space_gradients
./bootstrap.sh
```

### 3. `bootstrap.sh` - Environment Setup (Poetry)
**Best for**: Setting up environment in existing repository

Uses Poetry for dependency management:
- Creates virtual environment
- Installs dependencies via Poetry
- Handles HuggingFace auth
- Downloads model

**Usage**:
```bash
cd /path/to/input_space_gradients
./bootstrap.sh
```

### 4. `bootstrap_pip.sh` - Environment Setup (pip)
**Best for**: Fallback if Poetry has issues

Direct pip installation:
- Creates virtual environment
- Installs PyTorch with CUDA
- Installs all dependencies via pip
- Handles HuggingFace auth
- Downloads model

**Usage**:
```bash
cd /path/to/input_space_gradients
./bootstrap_pip.sh
```

## Configuration

All scripts use these defaults (edit as needed):

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `MOUNT_PATH` | `/mnt/data/llama-experiments` | Target directory for repository |
| `REPO_URL` | `git@github.com:kyle-pena-nlp/token-input-embedding-gradients.git` | GitHub repository |
| `REPO_NAME` | `input_space_gradients` | Directory name |
| `ENV_NAME` | `llama_epsilon_env` | Virtual environment name |

You can override `MOUNT_PATH` via environment variable:
```bash
export DEPLOY_MOUNT_PATH="/custom/path"
./setup_remote_complete.sh
```

## Typical Workflow

### First-time Setup
```bash
# 1. Copy the complete setup script
scp setup_remote_complete.sh user@gpu-machine:~/

# 2. SSH in and run
ssh user@gpu-machine
./setup_remote_complete.sh

# 3. Start working
cd /mnt/data/llama-experiments/input_space_gradients
source llama_epsilon_env/bin/activate
jupyter lab --no-browser
```

### Updating Code Later
```bash
ssh user@gpu-machine
cd /mnt/data/llama-experiments/input_space_gradients
git pull
source llama_epsilon_env/bin/activate
jupyter lab --no-browser
```

## SSH Setup for GitHub

The deployment scripts use SSH to clone from GitHub. Ensure your remote machine can access GitHub:

### Option 1: SSH Key Forwarding (Recommended)
```bash
# On local machine, edit ~/.ssh/config
Host gpu-machine
  HostName your-gpu-machine.com
  User youruser
  ForwardAgent yes

# Then just SSH normally
ssh gpu-machine
```

### Option 2: Add SSH Key on Remote
```bash
# On remote machine
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub

# Add this public key to GitHub:
# https://github.com/settings/keys
```

## Troubleshooting

### "Permission denied (publickey)" when cloning
- Set up SSH keys (see above)
- Or use HTTPS: edit `REPO_URL` to use `https://github.com/...`

### Mount path doesn't exist
```bash
sudo mkdir -p /mnt/data/llama-experiments
sudo chown $USER:$USER /mnt/data/llama-experiments
```

### Python version too old
Install Python 3.12+:
```bash
# Ubuntu/Debian
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv

# Or use conda
conda create -n py312 python=3.12
conda activate py312
```

### CUDA not found
Verify installation:
```bash
nvidia-smi          # Check GPU
nvcc --version      # Check CUDA toolkit
```

If missing, install CUDA toolkit for your distribution.

### Model download fails (401/403)
1. Accept license: https://huggingface.co/meta-llama/Llama-3.2-1B
2. Create token: https://huggingface.co/settings/tokens
3. Login: `huggingface-cli login`

## After Setup

Once setup is complete:

```bash
# Always activate the environment first
cd /mnt/data/llama-experiments/input_space_gradients
source llama_epsilon_env/bin/activate

# Start Jupyter
jupyter lab --no-browser --port=8888

# Or run scripts directly
python compute_batch_llama_epsilon_level_sets.py
```

Access Jupyter from your local machine:
```bash
# In a new terminal on your local machine
ssh -L 8888:localhost:8888 user@gpu-machine

# Then open http://localhost:8888 in your browser
```
