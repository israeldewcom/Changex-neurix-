#!/usr/bin/env bash
set -o errexit

echo "🔄 Starting ChangeX Neurix build process..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo "📥 Installing requirements..."
pip install --no-cache-dir -r requirements-render.txt

# Install spaCy model (small version for faster deployment)
echo "🤖 Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Alternative: Download larger model if needed
# python -m spacy download en_core_web_md

# Run database migrations
echo "🗄️ Running database migrations..."
if [ -f "migrations" ]; then
    flask db upgrade
else
    echo "No migrations found, skipping..."
fi

# Download additional AI models if needed
echo "⬇️ Downloading additional AI models..."
python -c "
try:
    from transformers import pipeline
    print('Downloading text generation model...')
    _ = pipeline('text-generation', model='gpt2')
except Exception as e:
    print(f'Model download skipped: {e}')
"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads/images uploads/videos uploads/audio static/generated

echo "✅ Build completed successfully!"
