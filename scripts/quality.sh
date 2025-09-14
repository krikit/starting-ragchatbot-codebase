#!/bin/bash

# Complete code quality check script
echo "🚀 Running complete code quality checks..."

echo "=================================="
echo "📦 Step 1: Import sorting (isort)"
echo "=================================="
uv run isort backend/ --check-only --diff
isort_exit=$?

echo "=================================="
echo "🖤 Step 2: Code formatting (black)"
echo "=================================="
uv run black backend/ --check --diff
black_exit=$?

echo "=================================="
echo "🔍 Step 3: Code linting (flake8)"
echo "=================================="
uv run flake8 backend/
flake8_exit=$?

echo "=================================="
echo "🔧 Step 4: Type checking (mypy)"
echo "=================================="
uv run mypy backend/
mypy_exit=$?

echo "=================================="
echo "📊 Quality Check Summary"
echo "=================================="

if [ $isort_exit -eq 0 ]; then
    echo "✅ isort: PASSED"
else
    echo "❌ isort: FAILED"
fi

if [ $black_exit -eq 0 ]; then
    echo "✅ black: PASSED"
else
    echo "❌ black: FAILED"
fi

if [ $flake8_exit -eq 0 ]; then
    echo "✅ flake8: PASSED"
else
    echo "❌ flake8: FAILED"
fi

if [ $mypy_exit -eq 0 ]; then
    echo "✅ mypy: PASSED"
else
    echo "❌ mypy: FAILED"
fi

# Exit with non-zero code if any tool failed
if [ $isort_exit -ne 0 ] || [ $black_exit -ne 0 ] || [ $flake8_exit -ne 0 ] || [ $mypy_exit -ne 0 ]; then
    echo ""
    echo "💥 Some quality checks failed. Run ./scripts/format-fix.sh to auto-fix formatting issues."
    exit 1
else
    echo ""
    echo "🎉 All quality checks passed!"
    exit 0
fi