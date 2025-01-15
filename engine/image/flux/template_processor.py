"""Template processor for FLUX workflows."""
import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape, Undefined


class WorkflowContextBuilder:
    """Builder for workflow template context."""

    @staticmethod
    def build_context(
        prompt: Union[str, Dict[str, Any]],
        workflow_config: Dict[str, Any],
        lora_config: Optional[Dict[str, Any]] = None,
        requires_lora: bool = False
    ) -> Dict[str, Any]:
        """Build context for workflow template from prompt and configurations."""
        # Handle prompt text
        prompt_text = prompt if isinstance(prompt, str) else prompt.get("text", "")

        # Extract workflow configuration with defaults
        context = {
            "prompt_text": prompt_text,
            "negative_text": workflow_config.get("negative_text", ""),
            "filename_prefix": workflow_config.get("filename_prefix", "output"),
            "width": workflow_config.get("width", 768),
            "height": workflow_config.get("height", 768),
            "steps": workflow_config.get("steps", 25),
            "cfg": workflow_config.get("cfg", 1.0),  # Default to 1.0 to match template
            "denoise": workflow_config.get("denoise", 1),
            "sampler_name": workflow_config.get("sampler", "euler"),
            "scheduler": workflow_config.get("scheduler", "normal"),
            "seed": workflow_config.get("seeds", [1])[0],  # Use first seed if multiple
            "ckpt_name": workflow_config.get("ckpt_name", "flux1-dev-fp8.safetensors")
        }

        # Handle LoRA configuration
        if requires_lora:
            if not lora_config:
                raise ValueError("No LoRA configuration found in YAML file. When using a custom workflow that requires a LoRA, you must specify the LoRA configuration in your YAML file.")

            # Ensure LoRA name has .safetensors extension
            lora_name = lora_config["name"]
            if not lora_name.endswith(".safetensors"):
                lora_name = f"{lora_name}.safetensors"

            # Set strength values
            strength = lora_config.get("strength", 0.6)
            context["lora"] = {
                "name": lora_name,
                "strength_model": strength,
                "strength_clip": strength
            }

        return context


class WorkflowTemplateProcessor(ABC):
    """Abstract base class for workflow template processors."""

    @abstractmethod
    def process_template(self, template_path: str, context: Dict[str, Any]) -> str:
        """Process the template with given context."""
        pass

    @abstractmethod
    def generate_workflow_path(self, output_dir: str, filename_prefix: str, context: Dict[str, Any]) -> str:
        """Generate the output path for the workflow file."""
        pass


class Jinja2TemplateProcessor(WorkflowTemplateProcessor):
    """Jinja2-based template processor for workflows."""

    def __init__(self):
        """Initialize the Jinja2 environment with custom filters and settings."""
        # Get the base directories for templates
        self.template_dir = str(Path(__file__).parent)
        self.project_root = str(Path(__file__).parent.parent.parent.parent)
        
        self.env = Environment(
            loader=FileSystemLoader([self.template_dir, self.project_root, "/"]),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
        self._add_custom_filters()

    def _add_custom_filters(self) -> None:
        """Add custom filters to the Jinja2 environment."""
        # Add json string filter that ensures proper JSON string escaping
        self.env.filters['json_str'] = self._json_str_filter
        
        # Add safe type conversion filters
        self.env.filters['safe_int'] = self._safe_int
        self.env.filters['safe_float'] = self._safe_float
        
        # Add LoRA name filter
        self.env.filters['lora_name'] = self._lora_name_filter

    @staticmethod
    def _lora_name_filter(name: str, default: str = "") -> str:
        """Convert LoRA name to proper format."""
        if not name:
            return default
        if not name.endswith(".safetensors"):
            name = f"{name}.safetensors"
        return name

    @staticmethod
    def _json_str_filter(obj: Any, default: str = "") -> str:
        """Convert value to JSON string with fallback for undefined values."""
        if isinstance(obj, Undefined):
            obj = default
        return json.dumps(obj)[1:-1]  # Remove outer quotes

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """Safely convert value to integer with fallback."""
        if isinstance(value, Undefined):
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert value to float with fallback."""
        if isinstance(value, Undefined):
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def process_template(self, template_path: str, context: Dict[str, Any]) -> str:
        """Process the Jinja2 template with given context."""
        try:
            template = self.env.get_template(template_path)
            
            # Ensure default values for LoRA configuration
            if 'lora' not in context:
                context['lora'] = {
                    'name': '',
                    'strength_model': 0.6,
                    'strength_clip': 0.6
                }
            
            # Render template with context
            rendered = template.render(**context)
            return rendered
        except Exception as e:
            logging.error(f"Error processing template: {e}")
            raise

    def generate_workflow_path(self, output_dir: str, filename_prefix: str, context: Dict[str, Any]) -> str:
        """Generate a unique workflow path based on context and datetime structure."""
        from datetime import datetime
        
        # Create a hash of the context to ensure unique filenames
        context_hash = hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()[:8]
        
        # Generate filename with prefix and hash
        filename = f"{filename_prefix}_{context_hash}.json"
        
        # Create datetime-based directory structure
        current_time = datetime.now()
        date_dir = current_time.strftime("%Y/%m/%d")
        time_dir = current_time.strftime("%H%M")
        
        # Combine all parts
        output_path = Path(output_dir) / date_dir / time_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Return full path
        return str(output_path / filename)
