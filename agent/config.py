from typing import List, Tuple, Dict, Any, Type
from pydantic import BaseModel, Field
from .cognitive_engine.models import CognitiveTrail
from .chat import BaseChatProvider
from .tools import BaseTool
from .embeddings import BaseEmbeddingProvider

class AgentConfig(BaseModel):
    CONTENT_SOURCES: List[Tuple[str, str]] = Field(default_factory=list)

class ContentProviderConfig(BaseModel):
    name: str
    inputs: Dict[str, Any]

class RetrieverConfig(BaseModel):
    ENABLED: bool = Field(default=False, description="Enable or disable retrieval feature.")
    NUM_RELEVANT_DOCUMENTS: int = Field(default=5)
    EMBEDDING_PROVIDER: Type[BaseEmbeddingProvider] = Field(description="Embedding provider class to use")
    EMBEDDING_PROVIDER_KWARGS: Dict[str, Any] = Field(default_factory=dict, description="Keyword arguments for the embedding provider")
    VECTOR_DB_NAME: str = Field(default="openai", description="Name of the vector database to use")
    VECTOR_DB_CONFIG: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configurations for different vector databases"
    )
    CONTENT_PROVIDERS: Dict[str, ContentProviderConfig] = Field(
        default_factory=dict,
        description="Configurations for content providers. Keys are provider IDs, values are ContentProviderConfig objects."
    )

class CognitiveEngineConfig(BaseModel):
    CHAT_PROVIDER: Type[BaseChatProvider] = Field(..., description="Chat provider class to use")
    CHAT_PROVIDER_KWARGS: Dict[str, Any] = Field(default_factory=dict, description="Keyword arguments for the chat provider")
    INITIAL_COGNITIVE_TRAIL: CognitiveTrail = Field(default_factory=CognitiveTrail)

class ToolConfig(BaseModel):
    TOOL_CLASS: Type[BaseTool] = Field(..., description="Tool class to use")
    TOOL_KWARGS: Dict[str, Any] = Field(default_factory=dict, description="Keyword arguments for the tool")

class ToolManagerConfig(BaseModel):
    ENABLED: bool = Field(default=False, description="Enable or disable tool usage.")
    TOOLS: Dict[str, ToolConfig] = Field(default_factory=dict, description="Dictionary of tool configurations")

class Config(BaseModel):
    AGENT: AgentConfig = Field(default_factory=AgentConfig)
    RETRIEVER: RetrieverConfig = Field(default_factory=RetrieverConfig)
    COGNITIVE_ENGINE: CognitiveEngineConfig = Field(default_factory=CognitiveEngineConfig)
    TOOL_MANAGER: ToolManagerConfig = Field(default_factory=ToolManagerConfig)
