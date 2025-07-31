#!/bin/bash

# ML Ticket Router - Quick Start Script

set -e

echo "🚀 ML Ticket Router - Quick Start"
echo "================================"

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "⚠️  Docker is not installed. Running in local mode only."
    DOCKER_AVAILABLE=false
else
    DOCKER_AVAILABLE=true
fi

# Create virtual environment
echo "🔧 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{raw,processed,models} logs

# Set up environment
echo "⚙️  Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file. Please update with your settings."
fi

# Generate synthetic data and train initial model
echo "🤖 Training initial model with synthetic data..."
python scripts/train_model.py --synthetic --n-samples 5000

# Start services
if [ "$DOCKER_AVAILABLE" = true ]; then
    echo "🐳 Starting services with Docker Compose..."
    docker-compose up -d redis
    sleep 5
else
    echo "⚠️  Please start Redis manually: redis-server"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the API server:"
echo "  source venv/bin/activate"
echo "  uvicorn src.api.main:app --reload --port 8000"
echo ""
echo "Or with Docker:"
echo "  docker-compose up -d"
echo ""
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔑 Demo API Key: demo-key-123"
echo ""
echo "Example API call:"
echo 'curl -X POST "http://localhost:8000/api/v1/route-ticket" \'
echo '  -H "X-API-Key: demo-key-123" \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"ticket_id": "TEST-001", "description": "Cannot login to my account"}'"'"