import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

import yaml

from .config_schema import FluxConfig, WorkflowConfig, WorkflowType
from .template_processor import Jinja2TemplateProcessor, WorkflowContextBuilder


class WorkflowManager:
    """Manages workflow configurations and operations."""

    def __init__(self, config_path: str):
        """Initialize workflow manager with configuration file path."""
        self.config_path = config_path
        self.config: Optional[FluxConfig] = None
        self.workflows: Optional[Dict[WorkflowType, WorkflowConfig]] = None
        self.logger = logging.getLogger(__name__)
        self.template_processor = Jinja2TemplateProcessor()
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

    def generate_workflow(self, template_path: str, output_dir: str, prompt: str) -> Tuple[str, dict]:
        """Generate a workflow file from template for a given prompt.
        
        Returns:
            Tuple[str, dict]: A tuple containing (workflow_path, workflow_metadata)
        """
        if not self.config or not self.workflow_config:
            raise RuntimeError("Configuration not loaded")

        # Get dev workflow configuration
        workflow_config = self.workflow_config.get('dev', {})
        lora_config = self.workflow_config.get('lora', [{}])[0]  # Get first LoRA config if exists

        # Build context for template
        context = WorkflowContextBuilder.build_context(
            prompt=prompt,
            workflow_config=workflow_config,
            lora_config=lora_config
        )

        # Process template and generate output path
        workflow_content = self.template_processor.process_template(template_path, context)
        output_path = self.template_processor.generate_workflow_path(
            output_dir=output_dir,
            filename_prefix=context['filename_prefix'],
            context=context
        )

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Write workflow file first
        with open(output_path, 'w') as f:
            f.write(workflow_content)
        self.logger.info(f"✨ Generated workflow file: {output_path}")

        # Create workflow metadata
        workflow_metadata = {
            "prompt": prompt,
            "template_path": template_path,
            "output_dir": output_dir,
            "workflow_config": workflow_config,
            "lora_config": lora_config,
            "context": {k: str(v) for k, v in context.items()},  # Convert all values to strings for JSON compatibility
            "timestamp": datetime.now().isoformat()
        }
        
        # Save workflow metadata as JSON alongside workflow file
        json_path = Path(output_path).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(workflow_metadata, f, indent=2)
        self.logger.info(f"💾 Saved workflow metadata to: {json_path}")

        return output_path, workflow_metadata
