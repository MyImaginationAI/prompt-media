import yaml


def test_prompt_config_structure(sample_prompt_config):
    """Test that the prompt config has all required sections"""
    assert "prompt_settings" in sample_prompt_config
    assert "prompts" in sample_prompt_config
    assert "workflows" in sample_prompt_config
    assert "variations" in sample_prompt_config


def test_workflow_config_validation(sample_prompt_config):
    """Test that workflow configurations have required parameters"""
    workflow = sample_prompt_config["workflows"]["test"]
    required_params = ["steps", "cfg_scale", "width", "height", "seeds"]

    for param in required_params:
        assert param in workflow, f"Missing required parameter: {param}"

    assert isinstance(workflow["steps"], int)
    assert isinstance(workflow["cfg_scale"], (int, float))
    assert isinstance(workflow["width"], int)
    assert isinstance(workflow["height"], int)
    assert isinstance(workflow["seeds"], list)


def test_prompt_settings_validation(sample_prompt_config):
    """Test that prompt settings are properly configured"""
    settings = sample_prompt_config["prompt_settings"]
    assert "prefix" in settings
    assert "negative" in settings
    assert isinstance(settings["prefix"], str)
    assert isinstance(settings["negative"], str)


def test_variations_validation(sample_prompt_config):
    """Test that variations are properly configured"""
    variations = sample_prompt_config["variations"]
    for var_name, var_config in variations.items():
        assert "type" in var_config
        assert "values" in var_config
        assert var_config["type"] in ["sequential", "random"]
        assert isinstance(var_config["values"], list)


def test_config_file_loading(temp_config_file):
    """Test that config file can be loaded correctly"""
    with open(temp_config_file) as f:
        loaded_config = yaml.safe_load(f)

    assert isinstance(loaded_config, dict)
    assert "prompt_settings" in loaded_config
    assert "workflows" in loaded_config
