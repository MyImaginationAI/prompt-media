import pytest
import yaml
from pathlib import Path
from engine.image.flux.run import main

def test_workflow_selection_with_loras(tmp_path):
    # Create a test prompt.yaml with loras
    prompt_yaml = {
        'prompt_settings': {
            'prefix': 'test',
            'negative': 'test negative'
        },
        'prompts': ['test prompt'],
        'loras': {
            'name': 'test_lora.safetensors',
            'strength_model': 1,
            'strength_clip': 1
        }
    }
    
    prompt_file = tmp_path / 'prompt.yaml'
    with open(prompt_file, 'w') as f:
        yaml.dump(prompt_yaml, f)

    # Test that the lora workflow is selected
    workflow_path = main(workflow_path='', prompt_media_path=str(prompt_file), dry_run=True)
    assert 'flux1-dev-fp8-1lora-api.json' in workflow_path

def test_workflow_selection_without_loras(tmp_path):
    # Create a test prompt.yaml without loras
    prompt_yaml = {
        'prompt_settings': {
            'prefix': 'test',
            'negative': 'test negative'
        },
        'prompts': ['test prompt']
    }
    
    prompt_file = tmp_path / 'prompt.yaml'
    with open(prompt_file, 'w') as f:
        yaml.dump(prompt_yaml, f)

    # Test that the default workflow is selected
    workflow_path = main(workflow_path='', prompt_media_path=str(prompt_file), dry_run=True)
    assert 'flux1-dev-fp8-api.json' in workflow_path

def test_workflow_selection_with_schnell(tmp_path):
    # Create a test prompt.yaml with schnell workflow
    prompt_yaml = {
        'prompt_settings': {
            'prefix': 'test',
            'negative': 'test negative'
        },
        'prompts': ['test prompt'],
        'workflows': {
            'schnell': {
                'steps': 4,
                'cfg_scale': 7
            }
        }
    }
    
    prompt_file = tmp_path / 'prompt.yaml'
    with open(prompt_file, 'w') as f:
        yaml.dump(prompt_yaml, f)

    # Test that the schnell workflow is selected
    workflow_path = main(workflow_path='schnell', prompt_media_path=str(prompt_file), dry_run=True)
    assert 'flux1-schnell-fp8-api.json' in workflow_path

def test_workflow_selection_with_lora_in_prompt(tmp_path):
    """Test that workflow with LoRA is selected when prompt contains LoRA tag."""
    prompt_yaml = {
        'prompt_settings': {
            'prefix': '',
            'negative': ''
        },
        'prompts': ['A beautiful landscape <lora:TestLora:0.6>']
    }
    
    prompt_file = tmp_path / 'prompt_with_lora.yaml'
    with open(prompt_file, 'w') as f:
        yaml.dump(prompt_yaml, f)

    workflow_path = main(workflow_path='', prompt_media_path=str(prompt_file), dry_run=True)
    assert 'flux1-dev-fp8-1lora-api.json' in workflow_path

def test_workflow_selection_with_empty_lora_tag(tmp_path):
    """Test that default workflow is selected when prompt contains empty LoRA tag."""
    prompt_yaml = {
        'prompt_settings': {
            'prefix': '',
            'negative': ''
        },
        'prompts': ['A beautiful landscape <lora:>']
    }
    
    prompt_file = tmp_path / 'prompt_with_empty_lora.yaml'
    with open(prompt_file, 'w') as f:
        yaml.dump(prompt_yaml, f)

    workflow_path = main(workflow_path='', prompt_media_path=str(prompt_file), dry_run=True)
    assert 'flux1-dev-fp8-api.json' in workflow_path

def test_workflow_selection_with_multiple_prompts_and_lora(tmp_path):
    """Test that LoRA workflow is selected when any prompt in the list contains LoRA."""
    prompt_yaml = {
        'prompt_settings': {
            'prefix': '',
            'negative': ''
        },
        'prompts': [
            'A beautiful landscape',
            'A sunset over mountains <lora:SunsetLora:0.8>',
            'A forest path'
        ]
    }
    
    prompt_file = tmp_path / 'prompt_with_multiple.yaml'
    with open(prompt_file, 'w') as f:
        yaml.dump(prompt_yaml, f)

    workflow_path = main(workflow_path='', prompt_media_path=str(prompt_file), dry_run=True)
    assert 'flux1-dev-fp8-1lora-api.json' in workflow_path
