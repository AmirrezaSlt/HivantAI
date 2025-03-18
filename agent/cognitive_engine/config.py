from pydantic import BaseModel, Field, InstanceOf
from typing import List
from .llm import BaseLLMProvider

class CognitiveEngineConfig(BaseModel):
    LLM_PROVIDER: InstanceOf[BaseLLMProvider] = Field(..., description="LLM provider")
    MAX_TOKENS: int = Field(default=1000, description="Maximum number of tokens")
    TEMPERATURE: float = Field(default=0.7, description="Temperature")
    MAX_ITERATIONS: int = Field(default=10, description="Maximum number of iterations")
    AGENT_NAME: str = Field(default="Agent", description="Name of the agent")
    AGENT_ROLE: str = Field(default="You are an AI assistant that thinks step by step.", description="Role of the agent")
    AGENT_PERMISSIONS: List[str] = Field(default=[], description="Permissions of the agent")
