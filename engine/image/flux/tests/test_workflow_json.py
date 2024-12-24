import json
import os
import pytest
from pathlib import Path
from engine.image.flux.workflow import WorkflowManager

@pytest.fixture
def workflow_manager(tmp_path):
    config_path = str(tmp_path / "test_config.yaml")
    with open(config_path, "w") as f:
        f.write("""
workflows:
  dev:
    steps: 20
    cfg: 7.5
    width: 512
    height: 512
  default:
    steps: 30
    cfg: 7.5
    width: 512
    height: 512
  schnell:
    steps: 10
    cfg: 7.5
    width: 512
    height: 512
prompts:
  - "test prompt"
prefix_prompt: "prefix"
negative_prompt: "negative"
ooms:
  - test_oom:
      key: "test"
      value: "test_value"
      type: "random"
        """)
    return WorkflowManager(config_path)

def test_save_workflow_json(workflow_manager, tmp_path):
    template_path = str(tmp_path / "template.txt")
    with open(template_path, "w") as f:
        f.write("test template")
    
    output_dir = str(tmp_path / "output")
    prompt = "test prompt"
    
    # Generate workflow and get the path
    workflow_path = workflow_manager.generate_workflow(template_path, output_dir, prompt)
    
    # Check if JSON file exists
    json_path = Path(workflow_path).with_suffix('.json')
    assert json_path.exists()
    
    # Print file contents for debugging
    print("\nWorkflow file contents:")
    with open(workflow_path) as f:
        print(f.read())
    
    print("\nJSON file contents:")
    with open(json_path) as f:
        print(f.read())
    
    # Verify JSON content
    with open(json_path) as f:
        workflow_data = json.load(f)
        assert "prompt" in workflow_data
        assert workflow_data["prompt"] == prompt
        assert "template_path" in workflow_data
        assert "output_dir" in workflow_data
        assert "timestamp" in workflow_data
