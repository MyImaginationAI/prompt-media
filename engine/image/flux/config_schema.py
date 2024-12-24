from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class WorkflowType(str, Enum):
    """Types of workflows supported by the system."""

    DEV = "dev"
    SCHNELL = "schnell"
    DEFAULT = "default"


class OOMConfig(BaseModel):
    """Order of Magnitude configuration for prompt variations."""

    key: str = Field(..., description="Key identifier for the OOM")
    value: str = Field(..., description="Value or pattern for the OOM")
    type: str = Field(..., description="Type of OOM (random or sequential)")

    @validator("type")
    def validate_type(cls, v):
        if v not in ["random", "sequential"]:
            raise ValueError('type must be either "random" or "sequential"')
        return v


class WorkflowConfig(BaseModel):
    """Configuration for a specific workflow type."""

    steps: int = Field(..., ge=1, description="Number of steps for image generation")
    cfg: float = Field(..., ge=1, description="Configuration scale factor")
    width: int = Field(..., ge=64, description="Image width in pixels")
    height: int = Field(..., ge=64, description="Image height in pixels")
    seeds: Optional[List[int]] = Field(default=None, description="List of seeds for reproducibility")


class FluxConfig(BaseModel):
    """Main configuration for the Flux image generation system."""

    prefix_prompt: str = Field(..., description="Prefix to be added to all prompts")
    negative_prompt: str = Field(..., description="Negative prompt for image generation")
    prompts: List[str] = Field(..., min_items=1, description="List of base prompts")
    ooms: List[Dict[str, OOMConfig]] = Field(..., description="List of OOM configurations")
    workflows: Dict[WorkflowType, WorkflowConfig] = Field(..., description="Workflow-specific configurations")

    class Config:
        use_enum_values = True

    @validator("workflows")
    def validate_workflows(cls, v):
        """Ensure all workflow types have configurations."""
        required_workflows = {wf.value for wf in WorkflowType}
        if not all(wf in v for wf in required_workflows):
            missing = required_workflows - set(v.keys())
            raise ValueError(f"Missing configurations for workflows: {missing}")
        return v
