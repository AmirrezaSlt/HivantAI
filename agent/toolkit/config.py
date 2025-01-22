from pydantic import BaseModel, Field, InstanceOf
from typing import List
from .tool import BaseTool

class ToolkitConfig(BaseModel):
    ENABLED: bool = Field(default=False, description="Enable or disable tool usage.")
    TOOLS: List[InstanceOf[BaseTool]] = Field(default_factory=list, description="List of tools")