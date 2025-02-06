from pydantic import BaseModel, Field, InstanceOf
from .llm import BaseLLMProvider

class CognitiveEngineConfig(BaseModel):
    SYSTEM_PROMPT: str = Field(default="You are an AI assistant that thinks step by step.", description="System prompt")
    LLM_PROVIDER: InstanceOf[BaseLLMProvider] = Field(..., description="LLM provider")