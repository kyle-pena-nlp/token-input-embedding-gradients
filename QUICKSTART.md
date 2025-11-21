# 🚀 Quick Start - Deploy to Remote GPU Machine

## One-Command Deployment (From Your Local Machine)

```bash
./deploy_from_local.sh
# Follow the prompts, it will handle everything
```

That's it! The script will:
1. Copy setup files to your remote machine
2. Run the complete setup (optional)
3. Create a Jupyter tunnel helper script

---

## Or Step-by-Step

### 1️⃣ Copy Setup Script to Remote

```bash
scp setup_remote_complete.sh user@your-gpu-machine:~/
```

### 2️⃣ SSH and Run Setup

```bash
ssh user@your-gpu-machine
./setup_remote_complete.sh
```

### 3️⃣ Start Jupyter

```bash
cd /mnt/data/llama-experiments/input_space_gradients
source llama_epsilon_env/bin/activate
jupyter lab --no-browser
```

### 4️⃣ Connect from Local Machine

```bash
# In a new terminal
ssh -L 8888:localhost:8888 user@your-gpu-machine
```

Then open the Jupyter URL in your browser.

---

## Configuration

Edit these in `setup_remote_complete.sh` before running:

- **MOUNT_PATH**: Where to clone the repo (default: `/mnt/data/llama-experiments`)
- **REPO_URL**: GitHub repository (default is already set)

---

## Prerequisites on Remote Machine

- ✅ Ubuntu/Linux with NVIDIA GPU
- ✅ Python 3.12+
- ✅ CUDA 12.x
- ✅ Git installed

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| SSH fails | Set up SSH keys or use password auth |
| Python too old | Install Python 3.12: `sudo apt install python3.12` |
| CUDA not found | Install CUDA toolkit for your OS |
| Git authentication fails | Use SSH key forwarding or add SSH key to remote |
| Model download 401 | Accept license at https://huggingface.co/meta-llama/Llama-3.2-1B |

---

## What Gets Installed?

- 🐍 Python virtual environment
- 🔥 PyTorch 2.7.0 with CUDA support
- 🤗 Transformers + HuggingFace ecosystem
- 📊 Jupyter Lab + all analysis libraries
- 🦙 Llama-3.2-1B model (~5GB download)

---

## After Setup

```bash
# Always activate environment first
cd /mnt/data/llama-experiments/input_space_gradients
source llama_epsilon_env/bin/activate

# Open your notebook
jupyter lab --no-browser
```

---

## Need More Details?

- **Full guide**: See `SETUP.md`
- **Deployment scripts**: See `DEPLOYMENT.md`
- **Troubleshooting**: See `SETUP.md` → Troubleshooting section

---

## Model Cache Location

Models are cached in: `~/.cache/huggingface/hub/`

After first successful download, you can set `LOCAL_ONLY = True` in `llama_models.py` to avoid network calls.

---

## Recommended: Set up SSH Config

Add to `~/.ssh/config` on your local machine:

```ssh-config
Host gpu
    HostName your-gpu-machine.com
    User youruser
    ForwardAgent yes
    LocalForward 8888 localhost:8888
```

Then just: `ssh gpu`
