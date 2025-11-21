#!/bin/bash
set -e

echo "=================================================="
echo "Deploy to Remote Machine from Local"
echo "=================================================="
echo ""

# Configuration - EDIT THESE
REMOTE_HOST="${REMOTE_HOST:-}"  # e.g., user@gpu-machine.com
MOUNT_PATH="${DEPLOY_MOUNT_PATH:-/mnt/data/llama-experiments}"

# Prompt for remote host if not set
if [ -z "$REMOTE_HOST" ]; then
    read -p "Enter remote host (user@hostname): " REMOTE_HOST
fi

if [ -z "$REMOTE_HOST" ]; then
    echo "❌ ERROR: Remote host is required"
    exit 1
fi

echo "Remote host: $REMOTE_HOST"
echo "Target path: $MOUNT_PATH"
echo ""

# Test SSH connection
echo "Testing SSH connection..."
if ! ssh -o ConnectTimeout=5 "$REMOTE_HOST" "echo 'Connected'" &> /dev/null; then
    echo "❌ ERROR: Cannot connect to $REMOTE_HOST"
    echo "Make sure:"
    echo "  1. SSH is configured correctly"
    echo "  2. You can access the host: ssh $REMOTE_HOST"
    exit 1
fi
echo "✓ SSH connection successful"

# Copy setup script
echo ""
echo "Copying setup script to remote machine..."
scp setup_remote_complete.sh "$REMOTE_HOST:~/setup_remote_complete.sh"
echo "✓ Script copied"

# Update MOUNT_PATH in the script if needed
if [ "$MOUNT_PATH" != "/mnt/data/llama-experiments" ]; then
    echo "Updating mount path in script..."
    ssh "$REMOTE_HOST" "sed -i 's|^MOUNT_PATH=\".*\"|MOUNT_PATH=\"$MOUNT_PATH\"|' ~/setup_remote_complete.sh"
fi

# Make executable
ssh "$REMOTE_HOST" "chmod +x ~/setup_remote_complete.sh"

echo ""
echo "=================================================="
echo "Ready to Deploy"
echo "=================================================="
echo ""
echo "The setup script has been copied to:"
echo "  $REMOTE_HOST:~/setup_remote_complete.sh"
echo ""
echo "Choose how to proceed:"
echo ""
echo "Option 1 - Interactive (Recommended):"
echo "  ssh $REMOTE_HOST"
echo "  ./setup_remote_complete.sh"
echo ""
echo "Option 2 - Automatic (runs now):"
read -p "Run setup script automatically now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting remote setup..."
    echo "=================================================="

    # Run the setup script remotely
    ssh -t "$REMOTE_HOST" "bash ~/setup_remote_complete.sh"

    SETUP_STATUS=$?

    echo ""
    echo "=================================================="
    if [ $SETUP_STATUS -eq 0 ]; then
        echo "✓ Setup completed successfully!"
        echo ""
        echo "To start Jupyter Lab:"
        echo "  ssh -L 8888:localhost:8888 $REMOTE_HOST"
        echo "  cd $MOUNT_PATH/input_space_gradients"
        echo "  source llama_epsilon_env/bin/activate"
        echo "  jupyter lab --no-browser"
    else
        echo "❌ Setup failed with exit code $SETUP_STATUS"
        echo "You may need to SSH in and debug manually"
    fi
else
    echo ""
    echo "Skipping automatic execution."
    echo "SSH in and run the script manually:"
    echo "  ssh $REMOTE_HOST"
    echo "  ./setup_remote_complete.sh"
fi

echo ""
echo "=================================================="
echo ""

# Create a helper script for SSH tunneling
cat > /tmp/jupyter_tunnel.sh << EOF
#!/bin/bash
# Helper script to SSH with Jupyter port forwarding
echo "Connecting to $REMOTE_HOST with Jupyter tunnel..."
echo "After connecting, run:"
echo "  cd $MOUNT_PATH/input_space_gradients"
echo "  source llama_epsilon_env/bin/activate"
echo "  jupyter lab --no-browser"
echo ""
ssh -L 8888:localhost:8888 "$REMOTE_HOST"
EOF

chmod +x /tmp/jupyter_tunnel.sh

echo "💡 Tip: A helper script has been created at /tmp/jupyter_tunnel.sh"
echo "   Run it to SSH with Jupyter port forwarding already set up"
echo ""
