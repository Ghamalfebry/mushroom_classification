#!/bin/bash

# Create necessary directories
mkdir -p examples

# Initialize Git repository
git init

# Configure Git (update with your details)
# echo "Enter your Git username:"
# read username
# echo "Enter your Git email:"
# read email

# git config user.name "$username"
# git config user.email "$email"

# Stage all files
git add .

# Create initial commit
git commit -m "Initial commit: Mushroom Classification Project"

# Set up remote repository
echo "Enter your GitHub repository URL (e.g., https://github.com/username/repo.git):"
read repo_url

git remote add origin $repo_url

# Push to GitHub
echo "Pushing to remote repository..."
git push -u origin master

echo "Git setup complete!"
