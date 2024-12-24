"""Integration tests for workflow template processing."""
import json
import os
from pathlib import Path

import pytest
import yaml

from ..template_processor import Jinja2TemplateProcessor, WorkflowContextBuilder
from ..workflow import WorkflowManager


@pytest.fixture
def workspace_root():
    """Get the workspace root directory."""
    return Path(os.environ.get("WORKSPACE", Path(__file__).parent.parent.parent.parent.parent))


@pytest.fixture
def sample_workflow_path(workspace_root):
    """Get the path to the sample workflow file."""
    return workspace_root / "engine/image/flux/workflows/api/flux1-dev-fp8-1lora-api.json"


@pytest.fixture
def sample_prompt_path(workspace_root):
    """Get the path to the sample prompt file."""
    return workspace_root / "collections/prompts/loras/luminous-shadowscape.yaml"


def test_workflow_generation_with_lora(workspace_root, sample_workflow_path, sample_prompt_path):
    """Test generating a workflow file with LoRA configuration."""
    # Load the prompt file
    with open(sample_prompt_path) as f:
        prompt_data = yaml.safe_load(f)

    # Initialize workflow manager
    manager = WorkflowManager(str(sample_prompt_path))

    # Create a temporary output directory
    output_dir = workspace_root / "tmp/test_workflows"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate workflow for the first prompt
    first_prompt = prompt_data["prompts"][0]
    workflow_path = manager.generate_workflow(
        template_path=str(sample_workflow_path),
        output_dir=str(output_dir),
        prompt=first_prompt
    )

    # Verify the generated workflow
    assert Path(workflow_path).exists()
    with open(workflow_path) as f:
        workflow = json.load(f)

    # Verify workflow content
    assert "inputs" in workflow["6"]  # CLIP Text Encode node
    assert workflow["6"]["inputs"]["text"] == first_prompt
    assert workflow["9"]["inputs"]["filename_prefix"] == prompt_data["workflows"]["filename_prefix"]

    # Verify LoRA configuration if present
    if "loras" in prompt_data and prompt_data["loras"].get("lora"):
        lora_config = prompt_data["loras"]["lora"][0]
        # Add verification for LoRA-specific nodes in the workflow

    # Clean up
    Path(workflow_path).unlink()
    output_dir.rmdir()
