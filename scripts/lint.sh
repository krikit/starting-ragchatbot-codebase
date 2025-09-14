#!/bin/bash

# Code linting script
echo "Running code linting..."

echo "🔍 Running flake8..."
uv run flake8 backend/

echo "🔧 Running mypy..."
uv run mypy backend/

echo "✅ Code linting complete!"