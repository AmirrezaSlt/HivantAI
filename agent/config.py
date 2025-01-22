from typing import Dict, Any, Type, List
from pydantic import BaseModel, Field, InstanceOf
from .embeddings import BaseEmbeddingProvider
from .vector_db import BaseVectorDB
from .reference_documents import BaseReferenceDocument
from .llm import BaseLLMProvider
from .tools import BaseTool

class RetrieverConfig(BaseModel):
    ENABLED: bool = Field(default=False, description="Enable or disable retrieval feature.")
    NUM_RELEVANT_DOCUMENTS: int = Field(default=3)
    EMBEDDING_PROVIDER: InstanceOf[BaseEmbeddingProvider] = Field(description="Embedding provider")
    VECTOR_DB: InstanceOf[BaseVectorDB] = Field(description="Vector database")
    REFERENCE_DOCUMENTS: List[InstanceOf[BaseReferenceDocument]] = Field(
        default_factory=list,
        description="List of reference documents for RAG"
    )

class CognitiveEngineConfig(BaseModel):
    LLM_PROVIDER: Type[BaseLLMProvider] = Field(..., description="LLM provider class to use")
    LLM_PROVIDER_KWARGS: Dict[str, Any] = Field(default_factory=dict, description="Keyword arguments for the LLM provider")

class ToolConfig(BaseModel):
    TOOL_CLASS: Type[BaseTool] = Field(..., description="Tool class to use")
    TOOL_KWARGS: Dict[str, Any] = Field(default_factory=dict, description="Keyword arguments for the tool")

class ToolManagerConfig(BaseModel):
    ENABLED: bool = Field(default=False, description="Enable or disable tool usage.")
    TOOLS: Dict[str, ToolConfig] = Field(default_factory=dict, description="Dictionary of tool configurations")

class Config(BaseModel):
    RETRIEVER: RetrieverConfig = Field(default_factory=RetrieverConfig)
    COGNITIVE_ENGINE: CognitiveEngineConfig = Field(default_factory=CognitiveEngineConfig)
    TOOL_MANAGER: ToolManagerConfig = Field(default_factory=ToolManagerConfig)
# TODO: move to instances instead of classes and kwargs