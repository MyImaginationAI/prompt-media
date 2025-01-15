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
    from datetime import datetime
    from unittest.mock import patch
    import shutil

    # Mock current time to ensure consistent test results
    mock_time = datetime(2025, 1, 15, 15, 39)
    
    with patch('datetime.datetime') as mock_datetime:
        mock_datetime.now.return_value = mock_time
        
        # Load the prompt file
        with open(sample_prompt_path) as f:
            prompt_data = yaml.safe_load(f)

        # Initialize workflow manager
        manager = WorkflowManager(str(sample_prompt_path))

        # Create a temporary output directory
        output_dir = workspace_root / "tmp/test_workflows"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Generate workflow for the first prompt
            first_prompt = prompt_data["prompts"][0]
            workflow_path = manager.generate_workflow(
                template_path=str(sample_workflow_path),
                output_dir=str(output_dir),
                prompt=first_prompt
            )

            # Verify the generated workflow exists in the correct directory
            workflow_path_obj = Path(workflow_path)
            assert workflow_path_obj.exists()
            
            # Verify directory structure (YYYY/MM/DD/HHMM)
            expected_dir = output_dir / "2025" / "01" / "15" / "1539"
            assert workflow_path_obj.parent == expected_dir
            
            # Verify workflow content
            with open(workflow_path) as f:
                workflow = json.load(f)

            assert "inputs" in workflow["6"]  # CLIP Text Encode node
            assert workflow["6"]["inputs"]["text"] == first_prompt
            assert workflow["9"]["inputs"]["filename_prefix"] == prompt_data["workflows"]["filename_prefix"]

            # Verify metadata file exists
            metadata_path = workflow_path_obj.with_name(f"{workflow_path_obj.stem}_metadata.json")
            assert metadata_path.exists()

            # Verify LoRA configuration if present
            if "loras" in prompt_data and prompt_data["loras"].get("lora"):
                lora_config = prompt_data["loras"]["lora"][0]
                # Add verification for LoRA-specific nodes in the workflow

        finally:
            # Clean up entire directory structure
            if output_dir.exists():
                shutil.rmtree(output_dir)
