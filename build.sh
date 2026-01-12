#!/bin/bash
set -e

echo "🔧 Building ChangeX Neurix on Render..."
echo "Python version: $(python --version)"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install requirements
echo "Installing requirements..."
pip install -r requirements-render.txt

# Download spaCy model
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

echo "✅ Build successful!"
