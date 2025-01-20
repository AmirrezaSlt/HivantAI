from typing import List, Tuple, Dict, Any, Type
from pydantic import BaseModel, Field
from .llm import BaseLLMProvider
from .tools import BaseTool
from .embeddings import BaseEmbeddingProvider
from .vector_db import BaseVectorDB
from .data_sources import BaseDataSource

class RetrieverConfig(BaseModel):
    ENABLED: bool = Field(default=False, description="Enable or disable retrieval feature.")
    # NUM_RELEVANT_DOCUMENTS: int = Field(default=3)
    # EMBEDDING_PROVIDER: Type["BaseEmbeddingProvider"] = Field(description="Embedding provider class")
    # EMBEDDING_PROVIDER_KWARGS: Dict[str, Any] = Field(default_factory=dict, description="Keyword arguments for the embedding provider")
    # VECTOR_DB: Type["BaseVectorDB"] = Field(description="Vector database class")
    # VECTOR_DB_KWARGS: Dict[str, Any] = Field(
        # default_factory=dict,
        # description="Keyword arguments for the vector database"
    # )
    # DATA_SOURCES: Dict[str, "BaseDataSource"] = Field(
        # default_factory=dict, 
        # description="Dictionary of instantiated data sources"
    # )
# 
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
