import logging
from typing import Dict, List, Optional, Tuple

import yaml

from .config_schema import FluxConfig, WorkflowConfig, WorkflowType


class WorkflowManager:
    """Manages workflow configurations and operations."""

    def __init__(self, config_path: str):
        """Initialize workflow manager with configuration file path."""
        self.config_path = config_path
        self.config: Optional[FluxConfig] = None
        self.workflows: Optional[Dict[WorkflowType, WorkflowConfig]] = None
        self.logger = logging.getLogger(__name__)
        self._load_config()

    def _load_config(self) -> None:
        """Load and validate configuration from file."""
        try:
            # Load prompt configuration
            with open(self.config_path, "r") as f:
                config_data = yaml.safe_load(f)

            self.workflow_config = None
            if "workflows" in config_data:
                self.workflow_config = config_data["workflows"]

            self.config = FluxConfig(**config_data)
            self.logger.info("✅ Configuration loaded and validated successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to load configuration: {e}")
            raise

    def get_workflow_config(self, workflow_path: str) -> Tuple[WorkflowType, WorkflowConfig]:
        """Get workflow configuration based on workflow path."""
        if not self.config:
            raise RuntimeError("Configuration not loaded")

        # Determine workflow type from path
        workflow_type = (
            WorkflowType.DEV if "dev" in workflow_path else WorkflowType.SCHNELL if "schnell" in workflow_path else WorkflowType.DEFAULT
        )

        workflow_config = self.config.workflows[workflow_type]
        self.logger.info(f"⚙️ Using {workflow_type} workflow configuration with {workflow_config.steps} steps")

        return workflow_type, workflow_config

    def get_prompts(self) -> List[str]:
        """Get the list of base prompts."""
        if not self.config:
            raise RuntimeError("Configuration not loaded")
        return self.config.prompts

    def get_prefix_prompt(self) -> str:
        """Get the global prefix prompt."""
        if not self.config:
            raise RuntimeError("Configuration not loaded")
        return self.config.prefix_prompt

    def get_negative_prompt(self) -> str:
        """Get the global negative prompt."""
        if not self.config:
            raise RuntimeError("Configuration not loaded")
        return self.config.negative_prompt

    def get_ooms(self) -> List[Dict[str, dict]]:
        """Get the OOM configurations."""
        if not self.config:
            raise RuntimeError("Configuration not loaded")
        return [{"oom": oom.dict()} for oom_dict in self.config.ooms for oom in oom_dict.values()]
