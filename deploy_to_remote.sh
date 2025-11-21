#!/bin/bash
set -e

echo "=================================================="
echo "Deploy Repository to Remote Machine"
echo "=================================================="

# Configuration - EDIT THESE AS NEEDED
REPO_URL="git@github.com:kyle-pena-nlp/token-input-embedding-gradients.git"
MOUNT_PATH="/mnt/data/llama-experiments"  # Change this to your mounted drive path
REPO_NAME="input_space_gradients"

# Allow override via environment variables
MOUNT_PATH="${DEPLOY_MOUNT_PATH:-$MOUNT_PATH}"

echo "Repository: $REPO_URL"
echo "Target path: $MOUNT_PATH/$REPO_NAME"
echo ""

# Check if mount point exists
if [ ! -d "$MOUNT_PATH" ]; then
    echo "❌ ERROR: Mount path $MOUNT_PATH does not exist"
    echo "Please create the directory or update MOUNT_PATH in this script"
    exit 1
fi

# Check if we can write to the mount point
if [ ! -w "$MOUNT_PATH" ]; then
    echo "❌ ERROR: Cannot write to $MOUNT_PATH"
    echo "Check permissions or run with appropriate privileges"
    exit 1
fi

echo "✓ Mount path is accessible"

# Full target path
TARGET_PATH="$MOUNT_PATH/$REPO_NAME"

# Check if repository already exists
if [ -d "$TARGET_PATH" ]; then
    echo "⚠ Repository already exists at $TARGET_PATH"
    read -p "Update existing repository? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Updating repository..."
        cd "$TARGET_PATH"

        # Check if it's a valid git repo
        if [ ! -d ".git" ]; then
            echo "❌ ERROR: Directory exists but is not a git repository"
            exit 1
        fi

        # Stash any local changes
        if ! git diff-index --quiet HEAD --; then
            echo "Stashing local changes..."
            git stash
        fi

        # Pull latest changes
        echo "Pulling latest changes..."
        git pull origin main || git pull origin master

        echo "✓ Repository updated"
    else
        echo "Using existing repository without updates"
    fi
else
    echo "Cloning repository..."

    # Check if git is installed
    if ! command -v git &> /dev/null; then
        echo "❌ ERROR: git is not installed"
        exit 1
    fi

    # Check SSH key access
    echo "Testing SSH access to GitHub..."
    if ! ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
        echo "⚠ WARNING: SSH authentication to GitHub may fail"
        echo "Make sure your SSH key is added to GitHub"
        echo "See: https://docs.github.com/en/authentication/connecting-to-github-with-ssh"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Clone the repository
    cd "$MOUNT_PATH"
    git clone "$REPO_URL" "$REPO_NAME"

    echo "✓ Repository cloned successfully"
    cd "$TARGET_PATH"
fi

# Show repository status
echo ""
echo "=================================================="
echo "Repository Status"
echo "=================================================="
cd "$TARGET_PATH"
echo "Location: $(pwd)"
echo "Current branch: $(git branch --show-current)"
echo "Latest commit: $(git log -1 --oneline)"
echo ""

# Check for required files
echo "Checking required files..."
REQUIRED_FILES=(
    "llama_models.py"
    "find_llama_epsilon_level_sets.ipynb"
    "compute_batch_llama_epsilon_level_sets.py"
    "pyproject.toml"
    "bootstrap.sh"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ❌ $file (missing)"
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo ""
    echo "⚠ WARNING: Some required files are missing"
    echo "This may not be the correct repository or branch"
fi

echo ""
echo "=================================================="
echo "Next Steps"
echo "=================================================="
echo ""
echo "1. Navigate to the repository:"
echo "   cd $TARGET_PATH"
echo ""
echo "2. Run the bootstrap script:"
echo "   ./bootstrap.sh"
echo ""
echo "3. Or run it directly:"
echo "   cd $TARGET_PATH && ./bootstrap.sh"
echo ""
echo "Repository deployed successfully!"
echo ""
