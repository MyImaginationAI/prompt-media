"""Tests for the template processor."""
import json
import pytest
from pathlib import Path

from engine.image.flux.template_processor import WorkflowContextBuilder, Jinja2TemplateProcessor


@pytest.fixture
def workflow_config():
    return {
        "negative_text": "bad quality, blurry",
        "filename_prefix": "test_output",
        "width": 768,
        "height": 768,
        "steps": 30,
        "cfg": 7,
        "denoise": 1,
        "sampler": "euler",
        "scheduler": "normal",
        "seeds": [1],
        "ckpt_name": "flux1-dev-fp8.safetensors"
    }


def test_build_context_with_lora_in_prompt():
    """Test building context with LoRA specified in prompt text."""
    prompt = "A test prompt"
    builder = WorkflowContextBuilder()
    context = builder.build_context(prompt, {})
    
    assert context["lora_name"] == "Luminous_Shadowscape-000016.safetensors"
    assert context["strength_model"] == 0.8
    assert context["strength_clip"] == 0.8
    assert context["prompt_text"] == prompt


def test_build_context_with_lora_config():
    """Test building context with LoRA specified in config."""
    lora_config = {
        "Luminous_Shadowscape-000016": {
            "strength_model": 0.7,
            "strength_clip": 0.7
        }
    }
    builder = WorkflowContextBuilder()
    context = builder.build_context("A test prompt", {}, lora_config)
    
    assert context["lora_name"] == "Luminous_Shadowscape-000016.safetensors"
    assert context["strength_model"] == 0.7
    assert context["strength_clip"] == 0.7


def test_build_context_with_invalid_lora():
    """Test building context with invalid LoRA name."""
    lora_config = {
        "InvalidLora": {
            "strength_model": 0.7,
            "strength_clip": 0.7
        }
    }
    builder = WorkflowContextBuilder()
    
    with pytest.raises(ValueError, match=r"Invalid LoRA name:.*"):
        builder.build_context("A test prompt", {}, lora_config)


def test_process_template(tmp_path):
    """Test processing a workflow template."""
    # Create a test template
    template_path = tmp_path / "test_template.json"
    template_content = {
        "test_node": {
            "inputs": {
                "text": "{{ prompt_text | json_str }}",
                "lora_name": "{{ lora_name | json_str }}"
            }
        }
    }
    template_path.write_text(json.dumps(template_content))
    
    # Process template
    processor = Jinja2TemplateProcessor()
    context = {
        "prompt_text": "Test prompt",
        "lora_name": "test_lora.safetensors"
    }
    
    result = processor.process_template(str(template_path), context)
    result_json = json.loads(result)
    
    assert result_json["test_node"]["inputs"]["text"] == "Test prompt"
    assert result_json["test_node"]["inputs"]["lora_name"] == "test_lora.safetensors"


def test_generate_workflow_path(tmp_path):
    """Test generating workflow path with datetime directory structure."""
    from datetime import datetime
    from unittest.mock import patch
    
    processor = Jinja2TemplateProcessor()
    context = {
        "prompt_text": "Test prompt",
        "lora_name": "test_lora.safetensors"
    }
    
    # Mock current time to ensure consistent test results
    mock_time = datetime(2025, 1, 15, 15, 39)
    with patch('datetime.datetime') as mock_datetime:
        mock_datetime.now.return_value = mock_time
        
        path = processor.generate_workflow_path(str(tmp_path), "test", context)
        path_obj = Path(path)
        
        # Verify file extension and name pattern
        assert path_obj.suffix == ".json"
        assert "test" in path_obj.name
        
        # Verify directory structure (YYYY/MM/DD/HHMM)
        expected_dir = tmp_path / "2025" / "01" / "15" / "1539"
        assert path_obj.parent == expected_dir
        assert expected_dir.exists()
