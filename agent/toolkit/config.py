from pydantic import BaseModel, Field, InstanceOf
from typing import Dict, List, Optional
from .tool import BaseTool

class PythonCodeExecutorConfig(BaseModel):
    """
    Configuration for the Python code executor.
    """
    id: str = Field(
        default="code_executor",
        description="The ID of the code executor"
    )
    base_image: str = Field(
        default="python:3.13.1-slim",
        description="Docker base image to use (e.g., 'python:3.11-slim')"
    )
    python_version: str = Field(
        default="3.13",
        description="Python version required"
    )
    python_packages: List[str] = Field(
        default_factory=list,
        description="List of Python packages to install (e.g., ['numpy', 'pandas>=2.0.0'])"
    )
    system_packages: List[str] = Field(
        default_factory=list,
        description="List of system packages to install (e.g., ['git', 'curl'])"
    )
    resource_requests: Dict[str, str] = Field(
        default_factory=lambda: {"cpu": "200m", "memory": "512Mi"},
        description="K8s resource requests (e.g., {'cpu': '200m', 'memory': '512Mi'})"
    )
    resource_limits: Dict[str, str] = Field(
        default_factory=lambda: {"cpu": "200m", "memory": "512Mi"},
        description="K8s resource limits (e.g., {'cpu': '200m', 'memory': '512Mi'})"
    )
    environment_variables: Dict[str, Dict] = Field(
        default_factory=dict,
        description="Environment variables with descriptions and secret flags"
    )
    volumes: List[Dict] = Field(
        default_factory=list,
        description="List of volume configurations"
    )


class ToolkitConfig(BaseModel):
    ENABLED: bool = Field(default=False, description="Enable or disable tool usage.")
    TOOLS: List[InstanceOf[BaseTool]] = Field(default_factory=list, description="List of tools")
    EXECUTOR: Optional[PythonCodeExecutorConfig] = Field(None, description="Python code executor configuration")
