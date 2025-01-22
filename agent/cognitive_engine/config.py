from pydantic import BaseModel, Field, InstanceOf
from ..llm import BaseLLMProvider

class CognitiveEngineConfig(BaseModel):
    LLM_PROVIDER: InstanceOf[BaseLLMProvider] = Field(..., description="LLM provider")
