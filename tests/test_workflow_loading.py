import json
import pytest
import yaml
from pathlib import Path
from engine.image.flux.run import load_workflow

def test_workflow_loading_with_loras(tmp_path):
    # Create a test workflow with placeholders
    workflow = {
        "1": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": "{{ 1lora_name }}",
                "strength_model": "{{ 1lora_strength_model }}",
                "strength_clip": "{{ 1lora_strength_clip }}"
            }
        }
    }
    
    workflow_file = tmp_path / 'test_workflow.json'
    with open(workflow_file, 'w') as f:
        json.dump(workflow, f)
    
    # Create test prompt data with lora info
    prompt_data = {
        'loras': {
            'name': 'test_lora.safetensors',
            'strength_model': 0.8,
            'strength_clip': 0.7
        }
    }
    
    # Load and verify workflow
    loaded_workflow = load_workflow(str(workflow_file), prompt_data)
    
    assert loaded_workflow['1']['inputs']['model'] == 'test_lora.safetensors'
    assert loaded_workflow['1']['inputs']['strength_model'] == '0.8'
    assert loaded_workflow['1']['inputs']['strength_clip'] == '0.7'

def test_workflow_loading_without_loras(tmp_path):
    # Create a test workflow without placeholders
    workflow = {
        "1": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 123,
                "steps": 20
            }
        }
    }
    
    workflow_file = tmp_path / 'test_workflow.json'
    with open(workflow_file, 'w') as f:
        json.dump(workflow, f)
    
    # Load workflow without prompt data
    loaded_workflow = load_workflow(str(workflow_file))
    
    # Verify workflow remains unchanged
    assert loaded_workflow == workflow

def test_workflow_loading_with_prompts(tmp_path):
    # Create a test workflow with prompt placeholders
    workflow = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "{{prompt}}"
            }
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "{{negative_prompt}}"
            }
        }
    }
    
    workflow_file = tmp_path / 'test_workflow.json'
    with open(workflow_file, 'w') as f:
        json.dump(workflow, f)
    
    # Load and verify workflow with prompts
    loaded_workflow = load_workflow(
        str(workflow_file),
        prompt="test prompt",
        negative_prompt="test negative prompt"
    )
    
    assert loaded_workflow['1']['inputs']['text'] == 'test prompt'
    assert loaded_workflow['2']['inputs']['text'] == 'test negative prompt'

def test_workflow_loading_with_prompts_and_loras(tmp_path):
    # Create a test workflow with both prompt and lora placeholders
    workflow = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "{{prompt}}"
            }
        },
        "2": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": "{{ 1lora_name }}",
                "strength_model": "{{ 1lora_strength_model }}",
                "strength_clip": "{{ 1lora_strength_clip }}"
            }
        }
    }
    
    workflow_file = tmp_path / 'test_workflow.json'
    with open(workflow_file, 'w') as f:
        json.dump(workflow, f)
    
    # Create test prompt data with lora info
    prompt_data = {
        'loras': {
            'name': 'test_lora.safetensors',
            'strength_model': 0.8,
            'strength_clip': 0.7
        }
    }
    
    # Load and verify workflow
    loaded_workflow = load_workflow(
        str(workflow_file),
        prompt_data=prompt_data,
        prompt="test prompt"
    )
    
    assert loaded_workflow['1']['inputs']['text'] == 'test prompt'
    assert loaded_workflow['2']['inputs']['model'] == 'test_lora.safetensors'
    assert loaded_workflow['2']['inputs']['strength_model'] == '0.8'
    assert loaded_workflow['2']['inputs']['strength_clip'] == '0.7'
