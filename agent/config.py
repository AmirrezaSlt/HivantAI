from typing import Dict, Any, Type, List
from pydantic import BaseModel, Field, InstanceOf
from .toolkit import Toolkit
from .retriever import Retriever
from .cognitive_engine import CognitiveEngine

class AgentConfig(BaseModel):
    RETRIEVER: InstanceOf[Retriever] = Field(..., description="Retriever component")
    COGNITIVE_ENGINE: InstanceOf[CognitiveEngine] = Field(default=None, description="Optional cognitive engine component")
    TOOLKIT: InstanceOf[Toolkit] = Field(default=None, description="Optional toolkit component")