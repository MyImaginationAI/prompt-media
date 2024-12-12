"""Integration tests for workflow template processing."""
import json
import os
from pathlib import Path

import pytest
import yaml

from engine.image.flux.template_processor import Jinja2TemplateProcessor, WorkflowContextBuilder


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


@pytest.fixture
def output_dir(workspace_root):
    """Create and return a temporary output directory."""
    output_dir = workspace_root / "tmp/test_workflows"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
    # Cleanup
    for file in output_dir.glob("*.json"):
        file.unlink()
    output_dir.rmdir()


@pytest.fixture
def template_processor():
    """Create a template processor instance."""
    return Jinja2TemplateProcessor()


def test_workflow_generation_with_real_files(
    workspace_root, sample_workflow_path, sample_prompt_path, output_dir, template_processor
):
    """Test generating a workflow file using real workflow template and prompt file."""
    # Load the prompt file
    with open(sample_prompt_path) as f:
        prompt_data = yaml.safe_load(f)

    # Get configurations
    workflow_config = prompt_data["workflows"]["dev"]
    lora_config = prompt_data["loras"]["lora"][0] if "loras" in prompt_data else None
    first_prompt = prompt_data["prompts"][0]

    # Build context
    context = WorkflowContextBuilder.build_context(
        prompt=first_prompt,
        workflow_config=workflow_config,
        lora_config=lora_config
    )

    # Generate workflow
    workflow_content = template_processor.process_template(
        str(sample_workflow_path),
        context
    )
    workflow_path = template_processor.generate_workflow_path(
        str(output_dir),
        context["filename_prefix"],
        context
    )

    # Save workflow
    with open(workflow_path, "w") as f:
        f.write(workflow_content)

    # Verify the generated workflow
    assert Path(workflow_path).exists()
    with open(workflow_path) as f:
        workflow = json.load(f)

    # Verify key workflow components
    assert workflow["6"]["inputs"]["text"] == first_prompt
    assert workflow["27"]["inputs"]["width"] == workflow_config["width"]
    assert workflow["27"]["inputs"]["height"] == workflow_config["height"]
    assert workflow["31"]["inputs"]["steps"] == workflow_config["steps"]
    assert workflow["31"]["inputs"]["cfg"] == workflow_config["cfg_scale"]
    assert workflow["31"]["inputs"]["seed"] == workflow_config["seeds"][0]

    # Verify LoRA configuration if present
    if lora_config:
        assert workflow["36"]["inputs"]["lora_name"] == lora_config["name"]
        assert workflow["36"]["inputs"]["strength_model"] == lora_config["strength_model"]
        assert workflow["36"]["inputs"]["strength_clip"] == lora_config["strength_clip"]


def test_workflow_generation_with_missing_values(
    workspace_root, sample_workflow_path, output_dir, template_processor
):
    """Test generating a workflow file with missing values (should use defaults)."""
    # Minimal workflow config
    workflow_config = {
        "filename_prefix": "test_minimal"
    }
    prompt = "Test prompt"

    # Build context
    context = WorkflowContextBuilder.build_context(
        prompt=prompt,
        workflow_config=workflow_config
    )

    # Generate workflow
    workflow_content = template_processor.process_template(
        str(sample_workflow_path),
        context
    )
    workflow_path = template_processor.generate_workflow_path(
        str(output_dir),
        context["filename_prefix"],
        context
    )

    # Save workflow
    with open(workflow_path, "w") as f:
        f.write(workflow_content)

    # Verify the generated workflow
    with open(workflow_path) as f:
        workflow = json.load(f)

    # Verify default values are used
    assert workflow["27"]["inputs"]["width"] == 768  # Default width
    assert workflow["27"]["inputs"]["height"] == 768  # Default height
    assert workflow["31"]["inputs"]["steps"] == 30  # Default steps
    assert workflow["31"]["inputs"]["cfg"] == 7  # Default cfg_scale
    assert workflow["31"]["inputs"]["sampler_name"] == "euler"  # Default sampler
    assert workflow["31"]["inputs"]["scheduler"] == "normal"  # Default scheduler


def test_multiple_workflow_generation(
    workspace_root, sample_workflow_path, sample_prompt_path, output_dir, template_processor
):
    """Test generating multiple workflow files from the same prompt file."""
    # Load the prompt file
    with open(sample_prompt_path) as f:
        prompt_data = yaml.safe_load(f)

    workflow_paths = []
    for prompt in prompt_data["prompts"]:
        # Build context for each prompt
        context = WorkflowContextBuilder.build_context(
            prompt=prompt,
            workflow_config=prompt_data["workflows"]["dev"],
            lora_config=prompt_data["loras"]["lora"][0] if "loras" in prompt_data else None
        )

        # Generate workflow
        workflow_content = template_processor.process_template(
            str(sample_workflow_path),
            context
        )
        workflow_path = template_processor.generate_workflow_path(
            str(output_dir),
            context["filename_prefix"],
            context
        )

        # Save workflow
        with open(workflow_path, "w") as f:
            f.write(workflow_content)
        workflow_paths.append(workflow_path)

    # Verify all workflows were generated
    assert len(workflow_paths) == len(prompt_data["prompts"])
    
    # Verify each workflow is unique
    workflow_contents = set()
    for path in workflow_paths:
        with open(path) as f:
            content = f.read()
            workflow_contents.add(content)
    
    # Each workflow should be unique due to different prompts
    assert len(workflow_contents) == len(workflow_paths)
