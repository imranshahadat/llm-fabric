# _____________________________________________________________________________
# Project5: AI Engineering Automation
# Python 3.12 | uv | VS Code | macOS / Linux
# _____________________________________________________________________________

.DEFAULT_GOAL := help

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON_VERSION ?= 3.12
UV ?= uv
PKG ?=
ENTRY ?= src/app/main.py

export PATH := $(HOME)/.local/bin:$(HOME)/.cargo/bin:$(PATH)

# _____________________________________________________________________________
# Help & Onboarding
# _____________________________________________________________________________

.PHONY: help

# Display available commands
help:
	@echo ""
	@echo "Project5: Production-Grade RAG Application"
	@echo "_________________________________________________________"
	@echo ""
	@echo "FIRST-TIME SETUP (Start Here):"
	@echo "  make setup                Installs uv, Python $(PYTHON_VERSION), .venv & dependencies"
	@echo ""
	@echo "DAILY WORKFLOW:"
	@echo "  make run                  Start the application ($(ENTRY))"
	@echo "  make format               Format code and fix lint issues with Ruff"
	@echo "  make lint                 Check formatting and lint rules with Ruff"
	@echo "  make pyright              Run static type checking (Strict Mode)"
	@echo "  make test                 Run the full test suite"
	@echo "  make check                Run format check, lint, pyright, and tests"
	@echo ""
	@echo "UTILITIES:"
	@echo "  make add PKG=openai       Add a dependency"
	@echo "  make remove PKG=openai    Remove a dependency"
	@echo "  make vscode               Open project in VS Code"
	@echo "  make verify               Verify your environment is healthy"
	@echo ""

# _____________________________________________________________________________
# Pre-Flight Checks
# _____________________________________________________________________________

.PHONY: check-project check-uv check-env

# Ensure project directory exists
check-project:
	@test -d "$(PROJECT_DIR)" || \
		(echo "ERROR: Project directory does not exist."; exit 1)

# Verify uv is installed
check-uv:
	@if ! command -v $(UV) >/dev/null 2>&1; then \
		echo "ERROR: 'uv' is not installed on your system."; \
		echo "TIP: Run 'make setup' to install uv and Python automatically."; \
		exit 1; \
	fi

# Confirm virtual environment is ready
check-env: check-project check-uv
	@test -d "$(PROJECT_DIR)/.venv" || \
		(echo "ERROR: Virtual environment not found."; \
		echo "TIP: Run 'make setup' to initialize the project."; \
		exit 1)

# _____________________________________________________________________________
# Project Setup
# _____________________________________________________________________________

.PHONY: setup

# Bootstrap local environment from scratch
setup: check-project
	@echo "Starting Project5 Bootstrap..."
	@if ! command -v $(UV) >/dev/null 2>&1; then \
		echo "Installing uv package manager..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@echo "Installing Python $(PYTHON_VERSION)..."
	@$(UV) python install $(PYTHON_VERSION)
	@cd "$(PROJECT_DIR)" && $(UV) python pin $(PYTHON_VERSION)
	@if [ ! -d "$(PROJECT_DIR)/.venv" ]; then \
		echo "Creating virtual environment..."; \
		cd "$(PROJECT_DIR)" && $(UV) venv --python $(PYTHON_VERSION); \
	else \
		echo "Existing .venv detected."; \
	fi
	@if [ -f "$(PROJECT_DIR)/pyproject.toml" ]; then \
		echo "Syncing dependencies from pyproject.toml..."; \
		cd "$(PROJECT_DIR)" && $(UV) sync; \
	else \
		echo "WARNING: No pyproject.toml found. Initializing bare project..."; \
		cd "$(PROJECT_DIR)" && $(UV) init --bare && $(UV) sync; \
	fi
	@$(MAKE) vscode-settings
	@$(MAKE) verify
	@echo ""
	@echo "Setup complete! You can now run 'make run'."

# _____________________________________________________________________________
# Execution & QA
# _____________________________________________________________________________

.PHONY: run format lint pyright test check verify vscode-settings vscode

# Run main application entry point
run: check-env
	cd "$(PROJECT_DIR)" && $(UV) run python "$(ENTRY)"

# Format code and auto-fix lint issues with Ruff
format: check-env
	@echo "Formatting code with Ruff..."
	cd "$(PROJECT_DIR)" && $(UV) run ruff check --fix . && $(UV) run ruff format .

# Verify formatting and lint rules without modifying files
lint: check-env
	@echo "Checking lint and formatting with Ruff..."
	cd "$(PROJECT_DIR)" && $(UV) run ruff check . && $(UV) run ruff format --check .

# Check static types
pyright: check-env
	@echo "Running Pyright (Strict Mode)..."
	cd "$(PROJECT_DIR)" && $(UV) run pyright

# Run pytest suite
test: check-env
	@echo "Running Test Suite..."
	cd "$(PROJECT_DIR)" && $(UV) run pytest tests/ -v

# Run all quality checks in sequence
check: lint pyright test

# Print active Python path and version
verify: check-env
	@echo ""
	@echo "Environment Verification:"
	@echo "   Python: $$(cd "$(PROJECT_DIR)" && $(UV) run python --version)"
	@echo "   Path:   $$(cd "$(PROJECT_DIR)" && $(UV) run python -c 'import sys; print(sys.executable)')"
	@echo ""

# Configure workspace settings for VS Code without launching it
vscode-settings: check-project
	@mkdir -p "$(PROJECT_DIR)/.vscode"
	@printf '%s\n' \
	'{' \
	'    "python.defaultInterpreterPath": "$${workspaceFolder}/.venv/bin/python",' \
	'    "python.analysis.typeCheckingMode": "strict",' \
	'    "editor.formatOnSave": true,' \
	'    "[python]": { "editor.defaultFormatter": "charliermarsh.ruff" }' \
	'}' > "$(PROJECT_DIR)/.vscode/settings.json"
	@echo "VS Code settings updated for Strict Mode."

# Configure workspace settings and launch VS Code
vscode: vscode-settings
	cd "$(PROJECT_DIR)" && code .

# _____________________________________________________________________________
# Dependency Management
# _____________________________________________________________________________

.PHONY: add remove

# Install and lock a package
add: check-env
	@test -n "$(PKG)" || (echo "Usage: make add PKG=requests"; exit 1)
	cd "$(PROJECT_DIR)" && $(UV) add $(PKG)

# Uninstall a package
# remove: check-env
# 	@test -n "$(PKG)" || (echo "Usage: make remove PKG=requests"; exit 1)
# 	cd "$(PROJECT_DIR)" && $(UV) remove $(PKG)