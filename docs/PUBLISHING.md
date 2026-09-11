# Publishing Instinct8 Agent

Complete guide to building and publishing Instinct8 Agent to PyPI.

## Prerequisites

- PyPI account (https://pypi.org/account/register/)
- PyPI API token (https://pypi.org/manage/account/token/)
- Python 3.9+
- Virtual environment (recommended on macOS)

## Quick Publish

```bash
# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install build twine

# Build
python -m build

# Publish
python -m twine upload --username __token__ --password pypi-YOUR_TOKEN dist/*
```

Or use the automated script:
```bash
./scripts/publish.sh
```

## Step-by-Step Publishing

### 1. Update Version

Edit `pyproject.toml`:
```toml
version = "0.X.Y"  # Increment version
```

### 2. Set Up Virtual Environment (macOS)

On macOS with Homebrew Python, you need a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install build twine
```

### 3. Build Distribution

```bash
python -m build
```

This creates:
- `dist/instinct8_agent-X.Y.Z.tar.gz` (source distribution)
- `dist/instinct8_agent-X.Y.Z-py3-none-any.whl` (wheel)

### 4. Test on TestPyPI First (Optional)

```bash
python -m twine upload --repository testpypi \
  --username __token__ \
  --password pypi-YOUR_TEST_TOKEN \
  dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ instinct8-agent
instinct8 --help
```

### 5. Publish to Production PyPI

```bash
python -m twine upload \
  --username __token__ \
  --password pypi-YOUR_TOKEN \
  dist/*
```

### 6. Verify

```bash
pip install instinct8-agent
instinct8 --help
instinct8-agent --help
instinct8 "Hello, what can you do?"
```

## PyPI API Token Setup

### Option 1: Command Line (Simplest)

```bash
python -m twine upload --username __token__ --password pypi-YOUR_TOKEN_HERE dist/*
```

### Option 2: Environment Variables

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR_TOKEN_HERE
python -m twine upload dist/*
```

### Option 3: .pypirc File

Create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_TOKEN_HERE
```

Set permissions: `chmod 600 ~/.pypirc`

### Getting Your PyPI Token

1. Create account at https://pypi.org/account/register/
2. Go to https://pypi.org/manage/account/token/
3. Click "Add API token"
4. Choose scope (project-specific recommended)
5. Copy the token immediately (starts with `pypi-`)
6. Use `__token__` as username, your token as password

## Publishing Checklist

- [ ] Version updated in `pyproject.toml`
- [ ] CHANGELOG.md updated
- [ ] Tests pass: `make test`
- [ ] Virtual environment set up
- [ ] Distribution built: `python -m build`
- [ ] (Optional) Tested on TestPyPI
- [ ] Published to PyPI: `twine upload dist/*`
- [ ] Verified installation: `pip install instinct8-agent`
- [ ] Commands work: `instinct8 --help`, `instinct8-agent --help`
- [ ] Imports work: `python3 -c "from selective_salience import Instinct8Agent"`

## Alternative: Install via pipx

```bash
pipx install instinct8-agent
```

Simpler than Homebrew and doesn't require maintaining a formula.

## Troubleshooting

**"Package already exists"**
- Increment version in `pyproject.toml`

**"Invalid credentials"**
- Username must be exactly `__token__` (with underscores)
- Password must include full token starting with `pypi-`

**"Command not found: python"**
- Use `python3` instead of `python`

**"Permission denied" on macOS**
- Use virtual environment (see Step 2 above)

## Security Best Practices

1. Use project-scoped tokens when possible
2. Never commit tokens to git
3. Use `.pypirc` or environment variables instead of command line
4. Rotate tokens periodically
5. Delete unused tokens from PyPI account
