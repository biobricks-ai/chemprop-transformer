#!/bin/bash

# SSH into the server
# Note: The heredoc approach with 'EOF' will work correctly with SSH since it preserves quotes and escaping
ssh -t ubuntu@150.136.44.218 << 'EOF'

# Install GitHub CLI if not already installed
if ! command -v gh &> /dev/null; then
    type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && sudo apt update \
    && sudo apt install gh -y
fi

# Login to GitHub CLI (will prompt for authentication)
gh auth login

# Create and cd into toxtrain directory
mkdir -p toxtrain
cd toxtrain

# Clone the repo and checkout lambdalabs branch
gh repo clone biobricks-ai/chemprop-transformer
cd chemprop-transformer
git checkout lambdalabs

# Install Miniconda if not already installed
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    
    # Initialize conda
    source $HOME/miniconda/bin/activate
    conda init bash
    
    # Reload shell to get conda working
    source ~/.bashrc
fi

# Install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements.txt

# mkdir the cache/pack_multitask_tensors directory
mkdir -p cache/pack_multitask_tensors

EOF

# SCP up cache/pack_multitask_tensors/*
# Note: SCP command works correctly since it's outside the SSH heredoc
scp -r cache/pack_multitask_tensors/* ubuntu@150.136.44.218:toxtrain/chemprop-transformer/cache/pack_multitask_tensors/
