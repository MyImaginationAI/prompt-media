# =============================================================================
# Prompt Media Generation Makefile
# =============================================================================

# --- Environment Setup ---
export WORKSPACE=$(shell pwd)
export ENGINE=$(WORKSPACE)/engine
export ENGINE_IMAGE=$(ENGINE)/image
export PYTHONPATH:=$(WORKSPACE)

# Model-specific paths
export FLUX_SCRIPTS=$(ENGINE_IMAGE)/flux
export SD_SCRIPTS=$(ENGINE_IMAGE)/stable-diffusion

# Include other makefiles
-include .env
-include .env.rocm

# --- Configuration Variables ---
PYTHON ?= .venv/bin/python3

# Input/Output Configuration
COLLECTIONS_DIR = $(WORKSPACE)/collections
MEDIA_BASE_DIR = $(COLLECTIONS_DIR)/$(shell date +%Y/%m/%d)/media
MEDIA_IMAGES_DIR = $(MEDIA_BASE_DIR)/images
MEDIA_VIDEOS_DIR = $(MEDIA_BASE_DIR)/videos
MEDIA_AUDIO_DIR = $(MEDIA_BASE_DIR)/audio
MEDIA_OTHER_DIR = $(MEDIA_BASE_DIR)/other

# Override media directories with environment variables if set
ifdef PROMPT_MEDIA_BASE_DIR
    MEDIA_BASE_DIR = $(PROMPT_MEDIA_BASE_DIR)
endif

# Server Configuration (from .env)
SERVER_ADDRESS ?= $(APP_SERVER_HOST):$(APP_SERVER_PORT)

# --- FLUX Model Commands ---
FLUX_CMD = $(PYTHON) $(FLUX_SCRIPTS)/run.py
PROMPT_MEDIA_FILE ?= prompt.yaml

.PHONY: flux flux/dev flux/schnell flux/dry-run

# Simple development target
flux: flux/dev

# Development workflow
flux/dev:
	@mkdir -p "$(MEDIA_IMAGES_DIR)"
	$(FLUX_CMD) \
		--dev \
		--prompt-media $(PROMPT_MEDIA_FILE) \
		--output-dir $(MEDIA_IMAGES_DIR) \
		--server $(SERVER_ADDRESS)

# Schnell workflow
flux/schnell:
	@mkdir -p "$(MEDIA_IMAGES_DIR)"
	$(FLUX_CMD) \
		--schnell \
		--prompt-media $(PROMPT_MEDIA_FILE) \
		--output-dir $(MEDIA_IMAGES_DIR) \
		--server $(SERVER_ADDRESS)

# Dry run (no image generation)
flux/dry-run:
	@mkdir -p "$(MEDIA_IMAGES_DIR)"
	$(FLUX_CMD) \
		--dev \
		--prompt-media $(PROMPT_MEDIA_FILE) \
		--output-dir $(MEDIA_IMAGES_DIR) \
		--server $(SERVER_ADDRESS) \
		--dry-run

# --- Stable Diffusion Commands ---
SD_CMD = $(PYTHON) $(SD_SCRIPTS)/run.py
SD_PROMPTS ?= $(SD_SCRIPTS)/prompts.yaml

.PHONY: sd sd/txt2img sd/img2img sd/dry-run

sd/txt2img:
	@echo "Stable Diffusion text-to-image not implemented yet"

sd/img2img:
	@echo "Stable Diffusion image-to-image not implemented yet"

sd/dry-run:
	@echo "Stable Diffusion dry-run not implemented yet"

sd: sd/txt2img

# --- ComfyUI Configuration ---
COMFYUI_DIR ?= /home/valter/Workspaces/github.com/comfyanonymous/ComfyUI
COMFYUI_VENV ?= $(COMFYUI_DIR)/venv/bin/activate

.PHONY: comfyui comfyui/start comfyui/kill kill-comfyui

comfyui: comfyui/start

comfyui/start:
	@echo "Starting ComfyUI server..."
	@cd $(COMFYUI_DIR) && \
		bash -c "source $(COMFYUI_VENV) && \
		python main.py --listen --port 7860"

comfyui/kill:
	@echo "Killing ComfyUI processes..."
	@pkill -f "python.*ComfyUI" || echo "No ComfyUI processes found"

# Alias for comfyui/kill
kill-comfyui:
	@echo "Killing ComfyUI processes..."
	@pkill -f "python.*ComfyUI" || echo "No ComfyUI processes found"

# --- Image Composition Commands ---
COMPOSER_CMD = $(PYTHON) tools/image_composer.py
GIF_DURATION ?= 500
COLLAGE_WIDTH ?= 2400
BORDER_SIZE ?= 2
ROW_SPACING ?= 2

.PHONY: compose compose/gif compose/collage compose/all

compose: compose/all

# Create GIF from images in a directory
compose/gif:
	@if [ -z "$(INPUT_DIR)" ]; then \
		echo "Please specify INPUT_DIR=path/to/images/directory"; \
		exit 1; \
	fi
	@mkdir -p "$$(dirname "$(or $(OUTPUT_GIF),$(INPUT_DIR)/output.gif)")"
	$(COMPOSER_CMD) \
		"$(INPUT_DIR)" \
		--gif-output "$(or $(OUTPUT_GIF),$(INPUT_DIR)/output.gif)" \
		--gif-duration $(GIF_DURATION)

# Create collage from images in a directory
compose/collage:
	@if [ -z "$(INPUT_DIR)" ]; then \
		echo "Please specify INPUT_DIR=path/to/images/directory"; \
		exit 1; \
	fi
	@mkdir -p "$$(dirname "$(or $(OUTPUT_COLLAGE),$(INPUT_DIR)/collage.jpg)")"
	$(COMPOSER_CMD) \
		"$(INPUT_DIR)" \
		--collage-output "$(or $(OUTPUT_COLLAGE),$(INPUT_DIR)/collage.jpg)" \
		--collage-width $(COLLAGE_WIDTH) \
		--border-size $(BORDER_SIZE) \
		--row-spacing $(ROW_SPACING)

# Create both GIF and collage
compose/all:
	@if [ -z "$(INPUT_DIR)" ]; then \
		echo "Please specify INPUT_DIR=path/to/images/directory"; \
		exit 1; \
	fi
	$(COMPOSER_CMD) \
		"$(INPUT_DIR)" \
		--gif-output "$(or $(OUTPUT_GIF),$(INPUT_DIR)/output.gif)" \
		--collage-output "$(or $(OUTPUT_COLLAGE),$(INPUT_DIR)/collage.jpg)" \
		--gif-duration $(GIF_DURATION) \
		--collage-width $(COLLAGE_WIDTH) \
		--border-size $(BORDER_SIZE) \
		--row-spacing $(ROW_SPACING)

# --- Batch Processing Commands ---
.PHONY: batch batch/dev batch/schnell batch/dry-run

# Process all prompt media files with development workflow
batch/dev:
	@echo "Processing all prompt media files with development workflow..."
	@$(PYTHON) process_prompts.py --dev

# Process all prompt media files with schnell workflow
batch/schnell:
	@echo "Processing all prompt media files with schnell workflow..."
	@$(PYTHON) process_prompts.py --schnell

# Dry run to see what would be processed
batch/dry-run:
	@echo "Dry run - showing what would be processed..."
	@$(PYTHON) process_prompts.py --schnell --dry-run

# Default batch target uses schnell workflow
batch: batch/schnell

# --- Testing Commands ---
.PHONY: test test-coverage test-watch test-fast lint install-dev format

# Run all tests
test:
	$(PYTHON) -m pytest

# Run tests with coverage report
test-coverage:
	$(PYTHON) -m pytest --cov=. --cov-report=html --cov-report=term-missing

# Run tests in watch mode (requires pytest-watch)
test-watch:
	$(PYTHON) -m pip install pytest-watch
	$(PYTHON) -m pytest_watch

# Run tests without slow markers
test-fast:
	$(PYTHON) -m pytest -m "not slow"

# Format code and fix common issues
format:
	$(PYTHON) -m pip install black isort autoflake autopep8
	@echo "Removing unused imports with autoflake..."
	$(PYTHON) -m autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables .
	@echo "Fixing PEP8 issues with autopep8..."
	$(PYTHON) -m autopep8 --in-place --recursive --aggressive --aggressive .
	@echo "Formatting with black..."
	$(PYTHON) -m black .
	@echo "Sorting imports with isort..."
	$(PYTHON) -m isort .

# Check code style without making changes
lint:
	$(PYTHON) -m pip install flake8 black isort
	@echo "Running flake8..."
	$(PYTHON) -m flake8
	@echo "Running black..."
	$(PYTHON) -m black --check .
	@echo "Running isort..."
	$(PYTHON) -m isort --check .

# Install all development dependencies
install-dev:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install flake8 black isort pytest-watch

# --- Helper Commands ---
.PHONY: help

help:
	@echo "Available targets:"
	@echo "  flux/dev     - Run development workflow"
	@echo "  flux/schnell - Run schnell workflow"
	@echo "  flux/dry-run - Run without generating images"
	@echo ""
	@echo "Configuration:"
	@echo "  Server: $(SERVER_ADDRESS)"
	@echo "  Prompts: $(PROMPT_MEDIA_FILE)"
	@echo "  Output: $(MEDIA_IMAGES_DIR)"