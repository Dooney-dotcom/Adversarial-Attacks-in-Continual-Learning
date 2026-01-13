#!/bin/bash

# NOTE: Assumes python and pip are installed

# 1. Virtual Environment Creation
if [ -d ".venv" ]; then
    echo "Virtual environment venv already exists."
else
    echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

# 2. Activation and Dependency Installation
echo "Environment Activation and Dependency Installation"
source .venv/bin/activate

# Upgrade pip to the latest version
pip install --upgrade pip

# Install dependencies from requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "WARNING: requirements.txt not found, installing packages manually."
    pip install torch torchvision numpy pandas matplotlib tqdm
fi

echo -e "${GREEN}=== Setup Completed! ===${NC}"