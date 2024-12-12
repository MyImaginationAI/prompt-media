"""Unit tests for workflow template processor."""
import json
import os
import tempfile
from pathlib import Path
from typing import Dict

import pytest

from ..template_processor import Jinja2TemplateProcessor, WorkflowContextBuilder


@pytest.fixture
def template_processor():
    """Create a template processor instance."""
    return Jinja2TemplateProcessor()


@pytest.fixture
def sample_template(tmp_path):
    """Create a sample workflow template."""
    template_content = """{
        "inputs": {
            "text": "{{ prompt_text }}",
            "width": {{ width }},
            "height": {{ height }}
        },
        "filename_prefix": "{{ filename_prefix }}"
    }"""
    template_path = tmp_path / "test_template.json"
    template_path.write_text(template_content)
    return str(template_path)


@pytest.fixture
def sample_context() -> Dict:
    """Create a sample context for template processing."""
    return {
        "prompt_text": "Test prompt",
        "width": 1024,
        "height": 768,
        "filename_prefix": "test_output"
    }


def test_template_processing(template_processor, sample_template, sample_context):
    """Test that template processing works correctly."""
    result = template_processor.process_template(sample_template, sample_context)
    parsed = json.loads(result)
    
    assert parsed["inputs"]["text"] == sample_context["prompt_text"]
    assert parsed["inputs"]["width"] == sample_context["width"]
    assert parsed["inputs"]["height"] == sample_context["height"]
    assert parsed["filename_prefix"] == sample_context["filename_prefix"]


def test_workflow_path_generation(template_processor, sample_context):
    """Test that workflow path generation creates unique paths."""
    output_dir = tempfile.mkdtemp()
    path1 = template_processor.generate_workflow_path(
        output_dir, sample_context["filename_prefix"], sample_context
    )
    
    # Modify context slightly
    modified_context = sample_context.copy()
    modified_context["width"] = 832
    path2 = template_processor.generate_workflow_path(
        output_dir, sample_context["filename_prefix"], modified_context
    )
    
    # Paths should be different due to different context hashes
    assert path1 != path2
    assert all(p.endswith(".workflow.json") for p in [path1, path2])
    assert all(os.path.dirname(p) == output_dir for p in [path1, path2])


def test_context_builder():
    """Test that context builder creates correct context."""
    prompt = "Test prompt"
    workflow_config = {
        "filename_prefix": "test",
        "width": 1024,
        "height": 768,
        "steps": 25,
        "cfg_scale": 7,
        "seeds": [1, 2, 3]
    }
    lora_config = {
        "name": "test_lora.safetensors",
        "strength_model": 0.8,
        "strength_clip": 0.8
    }
    
    context = WorkflowContextBuilder.build_context(
        prompt=prompt,
        workflow_config=workflow_config,
        lora_config=lora_config
    )
    
    # Verify all expected fields are present
    assert context["prompt_text"] == prompt
    assert context["filename_prefix"] == workflow_config["filename_prefix"]
    assert context["width"] == workflow_config["width"]
    assert context["height"] == workflow_config["height"]
    assert context["steps"] == workflow_config["steps"]
    assert context["cfg_scale"] == workflow_config["cfg_scale"]
    assert context["seeds"] == workflow_config["seeds"]
    
    # Verify LoRA config
    assert context["lora"]["name"] == lora_config["name"]
    assert context["lora"]["strength_model"] == lora_config["strength_model"]
    assert context["lora"]["strength_clip"] == lora_config["strength_clip"]


def test_context_builder_defaults():
    """Test that context builder provides correct defaults."""
    prompt = "Test prompt"
    workflow_config = {}  # Empty config to test defaults
    
    context = WorkflowContextBuilder.build_context(prompt, workflow_config)
    
    # Verify default values
    assert context["prompt_text"] == prompt
    assert context["filename_prefix"] == "output"
    assert context["width"] == 768
    assert context["height"] == 768
    assert context["steps"] == 30
    assert context["cfg_scale"] == 7
    assert context["seeds"] == [1]
    assert "lora" not in context  # No LoRA config provided
