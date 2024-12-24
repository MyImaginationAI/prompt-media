"""Configuration management module."""

import os
from pathlib import Path
from typing import Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# Load environment variables from .env.rocm if it exists
if Path(".env.rocm").exists():
    load_dotenv(".env.rocm")


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = Field(default_factory=lambda: os.getenv("APP_SERVER_HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: int(os.getenv("APP_SERVER_PORT", "8188")))
    debug: bool = Field(default=False)


class ModelConfig(BaseModel):
    """Model configuration."""

    torch_device: str = Field(default_factory=lambda: os.getenv("APP_MODEL_TORCH_DEVICE", "cuda"))
    torch_dtype: str = Field(default_factory=lambda: os.getenv("APP_MODEL_TORCH_DTYPE", "float16"))
    torch_compile: bool = Field(default=False)
    torch_blas_prefer_hipblaslt: int = Field(default_factory=lambda: int(os.getenv("TORCH_BLAS_PREFER_HIPBLASLT", "0")))
    hsa_override_gfx_version: str = Field(default_factory=lambda: os.getenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0"))


class PathsConfig(BaseModel):
    """Paths configuration."""

    prompt_media: str = Field(default_factory=lambda: os.getenv("APP_PATHS_PROMPT_MEDIA", "prompt.yaml"))
    output_dir: str = Field(default_factory=lambda: os.getenv("APP_PATHS_OUTPUT_DIR", "generated_images"))


class MediaConfig(BaseModel):
    """Media configuration."""

    base_dir: str = Field(default_factory=lambda: os.getenv("PROMPT_MEDIA_BASE_DIR", "media"))
    paths: Dict[str, str] = Field(default={"images": "images", "videos": "videos", "audio": "audio", "other": "other"})


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default_factory=lambda: os.getenv("APP_LOGGING_LEVEL", "INFO"))
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file: str = Field(default_factory=lambda: os.getenv("APP_LOGGING_FILE", "app.log"))


class ApiConfig(BaseModel):
    """API configuration."""

    timeout: int = Field(default_factory=lambda: int(os.getenv("APP_API_TIMEOUT", "30")))


class Config(BaseModel):
    """Main configuration."""

    env: str = Field(default_factory=lambda: os.getenv("ENV", "development"))
    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)

    @classmethod
    def load(cls, config_file: Optional[str] = None) -> "Config":
        """Load configuration from YAML file and environment variables.

        Environment variables take precedence over YAML configuration.
        """
        config_data = {}
        if config_file and Path(config_file).exists():
            with open(config_file) as f:
                config_data = yaml.safe_load(f)

        return cls(**config_data)


# Global configuration instance
config = Config.load("config/default.yaml")
