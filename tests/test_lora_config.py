import pytest
import json
from pathlib import Path
from engine.image.flux.run import load_workflow, has_lora_in_prompts, select_workflow

@pytest.fixture
def sample_prompt_data():
    return {
        "prompt_settings": {
            "prefix": "",
            "negative": ""
        },
        "prompts": [
            "An enigmatic fortune teller in a dimly lit room"
        ],
        "loras": {
            "name": "Luminous_Shadowscape-000016.safetensors",
            "strength_model": 0.6,
            "strength_clip": 0.6
        }
    }

@pytest.fixture
def sample_prompt_data_with_tag():
    return {
        "prompt_settings": {
            "prefix": "",
            "negative": ""
        },
        "prompts": [
            "An enigmatic fortune teller <lora:Luminous_Shadowscape:0.6>"
        ]
    }

def test_has_lora_in_prompts_with_config(sample_prompt_data):
    """Test LoRA detection with loras configuration block"""
    assert has_lora_in_prompts(sample_prompt_data) == True

def test_has_lora_in_prompts_with_tag(sample_prompt_data_with_tag):
    """Test LoRA detection with lora tag in prompt"""
    assert has_lora_in_prompts(sample_prompt_data_with_tag) == True

def test_has_lora_in_prompts_without_lora():
    """Test LoRA detection with no LoRA configuration"""
    data = {
        "prompts": ["A simple prompt without LoRA"]
    }
    assert has_lora_in_prompts(data) == False

def test_select_workflow_with_lora(sample_prompt_data):
    """Test workflow selection with LoRA configuration"""
    workflow_path = select_workflow(sample_prompt_data)
    assert "1lora-api.json" in workflow_path

def test_select_workflow_without_lora():
    """Test workflow selection without LoRA configuration"""
    data = {
        "prompts": ["A simple prompt without LoRA"]
    }
    workflow_path = select_workflow(data)
    assert "1lora-api.json" not in workflow_path

def test_load_workflow_with_lora(sample_prompt_data):
    """Test workflow loading with LoRA configuration"""
    # Get the base workflow path
    base_dir = Path(__file__).parent.parent / "engine" / "image" / "flux" / "workflows" / "api"
    workflow_path = str(base_dir / "flux1-dev-fp8-1lora-api.json")
    
    # Load and customize the workflow
    workflow = load_workflow(workflow_path, sample_prompt_data)
    
    # Convert workflow to string for easy checking
    workflow_str = json.dumps(workflow)
    
    # Verify LoRA values are properly replaced
    assert sample_prompt_data["loras"]["name"] in workflow_str
    assert str(sample_prompt_data["loras"]["strength_model"]) in workflow_str
    assert str(sample_prompt_data["loras"]["strength_clip"]) in workflow_str
