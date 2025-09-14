#!/bin/bash

# Auto-fix code formatting script
echo "Auto-fixing code formatting..."

echo "📦 Running isort..."
uv run isort backend/

echo "🖤 Running black..."
uv run black backend/

echo "✅ Code formatting complete!"