# Contributing to Enhanced World Model

Thank you for your interest in contributing to the Enhanced World Model project! This document provides guidelines and best practices for contributing.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Branch Naming Convention](#branch-naming-convention)
4. [Commit Message Guidelines](#commit-message-guidelines)
5. [Pull Request Process](#pull-request-process)
6. [Code Quality Standards](#code-quality-standards)
7. [Testing Requirements](#testing-requirements)
8. [Issue Management](#issue-management)

---

## Getting Started

### Prerequisites

- Python 3.11+
- PyTorch 2.8+
- Git
- uv package manager

### Initial Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/Larwive/Enhanced-World-Model.git
   cd Enhanced-World-Model
   ```

2. **Install dependencies with uv:**
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install project dependencies
   uv sync

   # Install development dependencies
   uv sync --extra dev
   ```

   Or with pip (fallback):
   ```bash
   pip install -e .
   pip install -e ".[dev]"
   ```

3. **Set up pre-commit hooks (required):**
   ```bash
   uv run pre-commit install
   ```

   Pre-commit will automatically run code quality checks before each commit:
   - Trailing whitespace removal
   - End-of-file fixing
   - YAML/TOML validation
   - Ruff linting and formatting
   - Mypy type checking

---

## Development Workflow

### 1. Find or Create an Issue

**Before starting any work:**
- Check if there's an existing issue for your proposed changes
- If not, create a new issue describing:
  - The problem you're solving
  - Your proposed solution
  - Any architectural implications

**Issue should be approved before starting work on significant features.**

### 2. Create a Branch

Always create a new branch from `main` (see [Branch Naming Convention](#branch-naming-convention)).

### 3. Make Changes

- Keep changes **small and focused**
- Maximum of **3-5 files changed per PR**
- Each PR should address **one specific issue**
- Write tests for new functionality

### 4. Test Your Changes

```bash
# Run all pre-commit checks manually (recommended before committing)
uv run pre-commit run --all-files

# Run type checking
uv run mypy src/

# Run linting and formatting
uv run ruff check src/
uv run ruff format src/

# Run tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_vision.py -v
```

### 5. Commit Your Changes

Follow [Commit Message Guidelines](#commit-message-guidelines).

### 6. Push and Create Pull Request

```bash
git push origin feature/GH-123/add-vq-vae
```

Then create a PR on GitHub (see [Pull Request Process](#pull-request-process)).

---

## Branch Naming Convention

Use descriptive branch names that reflect the work being done:

```text
feature/GH-XXXX/add-new-calculator-support
fix/GH-XXXX/authentication-timeout-issue
docs/GH-XXXX/improve-installation-guide
refactor/GH-XXXX/simplify-api-caller
chore/GH-XXXX/update-dependencies
test/GH-XXXX/add-controller-tests
```

### Branch Name Components

1. **Type prefix:**
   - `feature/` - New features or enhancements
   - `fix/` - Bug fixes
   - `docs/` - Documentation changes
   - `refactor/` - Code refactoring (no behavior change)
   - `test/` - Adding or updating tests
   - `chore/` - Maintenance tasks (dependencies, tooling)

2. **Issue reference:** `GH-XXXX` where XXXX is the GitHub Issue number

3. **Description:** Short, kebab-case description

### Examples

```bash
# Feature branches
git checkout -b feature/GH-001/implement-vq-vae
git checkout -b feature/GH-002/add-temporal-transformer

# Bug fix branches
git checkout -b fix/GH-015/dynamic-batch-size-handling
git checkout -b fix/GH-016/planning-shape-mismatch

# Documentation branches
git checkout -b docs/GH-020/add-architecture-guide

# Refactor branches
git checkout -b refactor/GH-025/separate-discrete-continuous-controllers
```

---

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Format

```text
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `test` - Adding or updating tests
- `refactor` - Code refactoring (no behavior change)
- `style` - Code style changes (formatting, missing semicolons, etc.)
- `chore` - Maintenance tasks (dependencies, build scripts, etc.)
- `perf` - Performance improvements

### Scope

The scope should indicate which component is affected:

- `vision` - Vision model (VQ-VAE)
- `memory` - Memory model (TemporalTransformer)
- `controller` - Controller (Discrete/Continuous)
- `world-model` - WorldModel orchestrator
- `training` - Training scripts (A2C, pretraining)
- `deps` - Dependencies
- `ci` - CI/CD configuration
- `tests` - Test infrastructure

### Examples

```bash
# Good commit messages
feat(vision): add VQ-VAE with EMA quantizer
fix(memory): handle dynamic batch size in sequence buffer
docs(readme): add installation instructions for CUDA
test(controller): add unit tests for discrete action sampling
refactor(controller): separate discrete and continuous implementations
chore(deps): update PyTorch to 2.1.0
perf(vision): optimize encoder convolution layers

# With body
feat(memory): implement TemporalTransformer

Replace MDN-RNN with Transformer-based temporal model.
Uses 4-layer encoder with 8 attention heads for better
long-range dependency modeling.

Refs: GH-002

# Breaking change
feat(controller)!: change controller API to require z_t and h_t

BREAKING CHANGE: Controller.forward() now requires both z_t and h_t
as separate arguments instead of concatenated input.

Refs: GH-003
```

### Rules

1. **Use imperative mood** in the description ("add" not "added")
2. **Don't capitalize** the first letter of description
3. **No period** at the end of description
4. **Keep description under 72 characters**
5. **Reference the issue** in the footer: `Refs: GH-XXX`
6. **Use body** for complex changes to explain "why" not "what"

---

## Pull Request Process

### PR Size Guidelines

**Keep PRs small and focused:**

- Maximum **3-5 files changed** per PR
- Maximum **300 lines changed** (excluding tests)
- Single **logical change** per PR
- Must address **one specific issue**

**If your change is large, split it into multiple PRs:**

```text
# Example: Implementing VQ-VAE
PR #1: Add VectorQuantizer class
PR #2: Add VQ-VAE encoder
PR #3: Add VQ-VAE decoder
PR #4: Integrate VQ-VAE into WorldModel
```

### Creating a Pull Request

1. **Title format:**
   ```text
   [GH-XXX] Type: Brief description

   Examples:
   [GH-001] feat(vision): Add VQ-VAE implementation
   [GH-015] fix(memory): Handle dynamic batch sizes
   ```

2. **PR Description template:**
   ```markdown
   ## Summary
   Brief description of what this PR does.

   ## Related Issue
   Closes #XXX

   ## Changes Made
   - Change 1
   - Change 2
   - Change 3

   ## Testing
   - [ ] Unit tests added/updated
   - [ ] All tests passing
   - [ ] Type checking passes (mypy)
   - [ ] Manual testing completed

   ## Checklist
   - [ ] Pre-commit hooks passing
   - [ ] Code follows project style guidelines (Ruff)
   - [ ] Type checking passes (mypy)
   - [ ] Documentation updated (if needed)
   - [ ] Commit messages follow conventions
   - [ ] PR is focused on single issue
   - [ ] No unnecessary files changed
   ```

3. **Link to issue:**
   - Use `Closes #XXX` or `Fixes #XXX` in description
   - GitHub will auto-close the issue when PR merges

4. **Request review:**
   - Assign appropriate reviewers
   - Be responsive to feedback

### PR Review Process

**For reviewers:**
- Check that PR addresses only one issue
- Verify tests are included
- Check code quality and style
- Ensure mypy passes
- Verify documentation is updated

**For contributors:**
- Address review feedback promptly
- Keep discussions focused on the code
- Be open to suggestions
- Update PR based on feedback

### Merging

- PRs require **at least 1 approval**
- All CI checks must pass
- Use **squash and merge** for clean history
- Delete branch after merge

---

## Code Quality Standards

### Type Hints

**All functions must have type hints:**

```python
# Good
def forward(self, z_t: torch.Tensor, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass through controller."""
    ...

# Bad
def forward(self, z_t, h_t):
    ...
```

### Code Quality Tools

We use **pre-commit hooks** to enforce code quality standards automatically. The following tools run on every commit:

#### Ruff (Linter and Formatter)

Ruff is a fast Python linter and formatter that replaces Black, isort, and Flake8.

**Configuration** (in `pyproject.toml`):
```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP"]
```

**Run manually:**
```bash
# Check for issues
uv run ruff check src/

# Fix issues automatically
uv run ruff check --fix src/

# Format code
uv run ruff format src/
```

#### Mypy (Type Checking)

We use **mypy in normal mode** (not strict) for type checking.

**Configuration** (in `pyproject.toml`):
```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
ignore_missing_imports = true

# Per-module options for gradual typing
[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

**Run manually:**
```bash
# Check entire codebase
uv run mypy src/

# Check specific file
uv run mypy src/vision/VQ_VAE.py

# Check with verbose output
uv run mypy --show-error-codes src/
```

**Common type checking issues:**

```python
# Missing return type
def get_action(self, state):  # Error: Missing return type
    ...

# Fix
def get_action(self, state: torch.Tensor) -> torch.Tensor:
    ...

# Implicit optional
def reset(self, state: torch.Tensor = None):  # Error: Implicit optional

# Fix
from typing import Optional
def reset(self, state: Optional[torch.Tensor] = None) -> None:
    ...
```

### Code Style

**We follow PEP 8 with some exceptions:**

- Line length: **100 characters** (not 79)
- Use **double quotes** for strings
- Use **4 spaces** for indentation (no tabs)

**Formatting is automated via Ruff:**
```bash
# Format code
uv run ruff format src/

# Check formatting without making changes
uv run ruff format --check src/

# The pre-commit hook will automatically format your code on commit
```

### Docstrings

Use **Google-style docstrings:**

```python
def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute action distribution and value estimate.

    Args:
        z_t: Latent observation (B, latent_dim)
        a_t: Previous action (B, action_dim)

    Returns:
        action: Sampled action (B, action_dim)
        log_prob: Log probability of action (B, 1)
        value: Value estimate (B, 1)

    Raises:
        ValueError: If z_t and a_t have mismatched batch sizes.
    """
    ...
```

### Imports

**Order imports:**
1. Standard library
2. Third-party (torch, numpy, etc.)
3. Local imports

```python
# Good
import os
from typing import Tuple, Optional

import torch
import torch.nn as nn
import numpy as np

from src.Model import Model
from src.utils import load_config

# Bad (mixed ordering)
import torch
import os
from src.Model import Model
import numpy as np
```

---

## Testing Requirements

### Test Coverage

**All new code must have tests:**

- **Unit tests** for individual components
- **Integration tests** for component interactions
- **Minimum 80% coverage** for new code

### Writing Tests

**Test file structure:**
```
tests/
├── test_vision.py
├── test_memory.py
├── test_controller.py
├── test_world_model.py
└── integration/
    └── test_training.py
```

**Example test:**
```python
import torch
import pytest
from src.vision.VQ_VAE import VQ_VAE

class TestVQVAE:
    """Test suite for VQ-VAE vision model."""

    def test_forward_pass(self):
        """Test that forward pass produces expected shapes."""
        model = VQ_VAE(
            input_shape=(3, 64, 64),
            hidden_dim=256,
            embed_dim=64,
            num_embed=512
        )

        x = torch.randn(4, 3, 64, 64)  # Batch of 4 images
        recon, vq_loss, z_q = model(x)

        assert recon.shape == (4, 3, 64, 64)
        assert vq_loss.shape == ()
        assert z_q.shape[0] == 4
        assert z_q.shape[1] == 64

    def test_codebook_utilization(self):
        """Test that codebook doesn't collapse."""
        model = VQ_VAE(...)
        # Test implementation
        ...
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_vision.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run with verbose output
pytest tests/ -v

# Run specific test
pytest tests/test_vision.py::TestVQVAE::test_forward_pass -v
```

### Test Requirements for PR

**Before submitting PR:**
- [ ] All existing tests pass
- [ ] New tests added for new functionality if applicable

---

## Issue Management

### Creating Issues

**Use issue templates when available.**

**Good issue structure:**
```markdown
## Problem
Clear description of the problem or feature request.

## Proposed Solution
How you plan to solve it.

## Alternatives Considered
Other approaches you've thought about.

## Additional Context
Any relevant information, screenshots, error messages.

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
```

### Issue Labels

- `bug` - Something isn't working
- `feature` - New feature or enhancement
- `documentation` - Documentation improvements
- `refactor` - Code refactoring
- `testing` - Testing-related
- `good-first-issue` - Good for newcomers
- `help-wanted` - Extra attention needed
- `blocked` - Blocked by another issue
- `wip` - Work in progress

### Referencing Issues

**In commits:**
```bash
git commit -m "feat(vision): add VQ-VAE encoder

Refs: GH-001"
```

**In PRs:**
```markdown
Closes #123
Fixes #456
Refs #789
```

---

## Development Best Practices

### 1. Keep Changes Atomic

Each commit should represent **one logical change:**

```bash
# Good - separate commits
git commit -m "feat(vision): add VQ-VAE encoder"
git commit -m "test(vision): add encoder tests"
git commit -m "docs(vision): document encoder architecture"

# Bad - everything in one commit
git commit -m "add vq-vae with tests and docs"
```

### 2. Test Before Committing

```bash
# Pre-commit checklist (automated via pre-commit hooks)
uv run pre-commit run --all-files  # Run all checks
uv run mypy src/                   # Type checking
uv run pytest tests/               # Tests
uv run ruff format src/            # Format code
uv run ruff check --fix src/       # Lint and fix issues

# Or simply commit - pre-commit hooks will run automatically
git commit -m "your message"
```

### 3. Keep PRs Focused

**One PR = One Issue = One Feature/Fix**

If you find yourself changing many unrelated files, split into multiple PRs.

### 4. Update Documentation

When adding features:
- Update docstrings
- Update README if needed
- Update ARCHITECTURE_DECISIONS.md if architectural

### 5. Communicate Early

- Open draft PR early for feedback
- Ask questions in issue comments
- Use PR comments for specific questions

---

## Component-Specific Guidelines

### Vision Model (VQ-VAE)

- Maintain codebook utilization > 50%
- Test reconstruction quality visually
- Ensure gradient flow through straight-through estimator

### Memory Model (TemporalTransformer)

- Test with variable sequence lengths
- Verify attention masking is correct
- Check memory buffer management

### Controller

- Test action bounds (discrete: valid indices, continuous: within limits)
- Verify log probability computation
- Test both policy and value heads

---

## Questions?

If you have questions:
1. Check existing issues and PRs
2. Review ARCHITECTURE_DECISIONS.md
3. Open a new issue with the `question` label
4. Reach out to maintainers

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
