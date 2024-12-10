import pytest
import yaml


@pytest.fixture
def sample_prompt_config():
    return {
        "prompt_settings": {"prefix": "test prefix", "negative": "test negative"},
        "prompts": ["test prompt"],
        "variations": {"time": {"type": "sequential", "values": ["morning", "night"]}},
        "workflows": {"test": {"steps": 10, "cfg_scale": 7, "width": 512, "height": 512, "seeds": [1]}},
    }


@pytest.fixture
def temp_config_file(tmp_path, sample_prompt_config):
    config_file = tmp_path / "test_prompt.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_prompt_config, f)
    return config_file


@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("APP_PATHS_PROMPT_MEDIA", "test_prompt.yaml")
    monkeypatch.setenv("APP_PATHS_OUTPUT_DIR", "test_output")
    monkeypatch.setenv("APP_SERVER_HOST", "localhost")
    monkeypatch.setenv("APP_SERVER_PORT", "8188")
