#!/bin/bash

# Code formatting script
echo "Running code formatting..."

echo "📦 Running isort..."
uv run isort backend/ --check-only --diff

echo "🖤 Running black..."
uv run black backend/ --check --diff

echo "✅ Code formatting check complete!"