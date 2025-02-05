#!/bin/bash

# Install Micromamba if not already installed
if ! command -v micromamba &> /dev/null
then
    echo "Micromamba not found. Installing..."
    curl -Ls https://micro.mamba.pm/install.sh | bash
    source ~/.bashrc
fi

# Initialize Micromamba if not initialized
eval "$(micromamba shell hook --shell bash)"

# Create or update the environment
if micromamba env list | grep -q "strawberry_detection"; then
    echo "Environment 'strawberry_detection' already exists. Updating..."
    micromamba env update -f environment.yml
else
    echo "Creating environment 'strawberry_detection'..."
    micromamba create -f environment.yml
fi

# Activate the environment
micromamba activate strawberry_detection

# Add Jupyter kernel to allow selecting this environment in Jupyter
python -m ipykernel install --user --name=strawberry_detection --display-name "Python (strawberry_detection)"

echo "✅ Setup complete! Use 'micromamba activate strawberry_detection' to start."
echo "To run Jupyter Notebook, use: 'jupyter notebook'"
