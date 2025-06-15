#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install core dependencies first
# echo "Installing core dependencies..."
# pip install numpy pillow

# Install remaining dependencies
echo "Installing remaining dependencies..."
pip install -r requirements.txt

# Run the application
echo "Starting Streamlit app..."
streamlit run app.py
