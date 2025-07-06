#!/bin/bash
# This script exports the GitHub token for use in devcontainer

# Extract the GitHub token from gh auth status
export GH_TOKEN=$(gh auth token 2>/dev/null)

# Start the devcontainer with the token
echo "Starting devcontainer with GitHub authentication..."
code .