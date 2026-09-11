#!/bin/bash
# Publishing script for Instinct8 Agent

set -e

echo "=== Instinct8 Agent Publishing Script ==="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Must run from project root directory"
    exit 1
fi

# Check if build tools are installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found"
    exit 1
fi

echo "Step 1: Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo ""
echo "Step 2: Installing build tools..."
pip install -q build twine

echo ""
echo "Step 3: Building distribution..."
python -m build

echo ""
echo "Step 3: Distribution files created in dist/:"
ls -lh dist/

echo ""
read -p "Do you want to upload to TestPyPI? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Uploading to TestPyPI..."
    echo "Enter your TestPyPI token when prompted:"
    python -m twine upload --repository testpypi dist/*
    echo ""
    echo "✅ Uploaded to TestPyPI!"
    echo "Test installation with:"
    echo "  pip3 install --index-url https://test.pypi.org/simple/ instinct8-agent"
fi

echo ""
read -p "Do you want to upload to production PyPI? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Uploading to PyPI..."
    echo "Enter your PyPI token when prompted:"
    python -m twine upload dist/*
    echo ""
    echo "✅ Uploaded to PyPI!"
    echo "Users can now install with:"
    echo "  pip install instinct8-agent"
fi

echo ""
echo "=== Publishing Complete ==="
