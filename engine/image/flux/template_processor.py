"""Template processor for FLUX workflows."""
import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape, Undefined


class WorkflowContextBuilder:
    """Builder for workflow template context."""

    @staticmethod
    def build_context(
        prompt: Union[str, Dict[str, Any]],
        workflow_config: Dict[str, Any],
        lora_config: Optional[Dict[str, Any]] = None
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
            "steps": workflow_config.get("steps", 30),
            "cfg": workflow_config.get("cfg_scale", 7),
            "denoise": workflow_config.get("denoise", 1),
            "sampler_name": workflow_config.get("sampler", "euler"),
            "scheduler": workflow_config.get("scheduler", "normal"),
            "seed": workflow_config.get("seeds", [1])[0],  # Use first seed if multiple
            "ckpt_name": workflow_config.get("ckpt_name", "flux1-dev-fp8.safetensors")
        }

        # List of allowed LoRA names from ComfyUI server
        allowed_loras = [
            "AntiBlur.safetensors",
            "CPA.safetensors",
            "EldritchCandids1.1.2.safetensors",
            "FLUX-daubrez-DB4RZ.safetensors",
            "FluxMythV2.safetensors",
            "Hyper-FLUX.1-dev-8steps-lora.safetensors",
            "Luminous_Shadowscape-000016.safetensors",
            "SameFace_Fix.safetensors",
            "aidmaFLUXpro1.1-FLUX-V0.1.safetensors",
            "ck-shadow-circuit-000021.safetensors",
            "detailed_v2_flux_ntc.safetensors",
            "flux_realism_lora.safetensors",
            "midjourney_whisper_flux_lora_v01.safetensors"
        ]
        
        # Extract LoRA name from prompt if present
        import re
        lora_match = re.search(r'<lora:([^:]+):([^>]+)>', prompt_text)
        if lora_match:
            lora_base = lora_match.group(1)
            lora_strength = float(lora_match.group(2))
            
            # Find matching LoRA from allowed list
            lora_name = next((l for l in allowed_loras if l.startswith(lora_base)), "")
            
            if lora_name:
                context.update({
                    "lora_name": lora_name,
                    "strength_model": lora_strength,
                    "strength_clip": lora_strength
                })
        else:
            # Use provided LoRA config if any
            if lora_config:
                # Get the first LoRA name and config from the dictionary
                if isinstance(lora_config, dict) and lora_config:
                    lora_name = next(iter(lora_config))
                    lora_settings = lora_config[lora_name]
                    
                    # Ensure LoRA name has .safetensors extension
                    if not lora_name.endswith(".safetensors"):
                        lora_name = f"{lora_name}.safetensors"
                    
                    # Ensure the name matches exactly what ComfyUI expects
                    if lora_name not in allowed_loras:
                        raise ValueError(f"Invalid LoRA name: {lora_name}. Must be one of: {', '.join(allowed_loras)}")
                    
                    context.update({
                        "lora_name": lora_name,
                        "strength_model": lora_settings.get("strength_model", 0.6),
                        "strength_clip": lora_settings.get("strength_clip", 0.6)
                    })

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
        if isinstance(name, Undefined):
            return default
        if not name:
            return default
        # Always add .safetensors extension
        if not name.endswith(".safetensors"):
            name = f"{name}.safetensors"
        # Ensure the name matches exactly what ComfyUI expects
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
            # Convert template path to absolute path
            template_path = str(Path(template_path).absolute())
            
            # Make path relative to project root if it's under project root
            if template_path.startswith(self.project_root):
                template_path = str(Path(template_path).relative_to(self.project_root))
            
            # Load template from file
            template = self.env.get_template(template_path)
            
            # Ensure default values for LoRA configuration
            if 'lora_name' not in context:
                context['lora_name'] = ''
                context['strength_model'] = 0.6
                context['strength_clip'] = 0.6
            
            # Create nested lora structure for backward compatibility
            context['lora'] = {
                'name': context['lora_name'],
                'strength_model': context['strength_model'],
                'strength_clip': context['strength_clip']
            }
            
            # Render template with context
            rendered = template.render(**context)
            
            # Validate JSON
            json.loads(rendered)
            return rendered
        except Exception as e:
            raise ValueError(f"Failed to process template: {str(e)}") from e

    def generate_workflow_path(self, output_dir: str, filename_prefix: str, context: Dict[str, Any]) -> str:
        """Generate a unique workflow path based on context."""
        # Create a hash of the context to ensure unique filenames
        context_hash = hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()[:8]
        
        # Generate filename with prefix and hash
        filename = f"{filename_prefix}_{context_hash}.json"
        
        # Return full path
        return str(Path(output_dir) / filename)
