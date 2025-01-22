from pydantic import BaseModel, Field, InstanceOf
from typing import List
from ..embeddings import BaseEmbeddingProvider
from ..vector_db import BaseVectorDB
from ..reference_documents import BaseReferenceDocument

class RetrieverConfig(BaseModel):
    ENABLED: bool = Field(default=False, description="Enable or disable retrieval feature.")
    NUM_RELEVANT_DOCUMENTS: int = Field(default=3)
    EMBEDDING_PROVIDER: InstanceOf[BaseEmbeddingProvider] = Field(description="Embedding provider")
    VECTOR_DB: InstanceOf[BaseVectorDB] = Field(description="Vector database")
    REFERENCE_DOCUMENTS: List[InstanceOf[BaseReferenceDocument]] = Field(
        default_factory=list,
        description="List of reference documents for RAG"
    )