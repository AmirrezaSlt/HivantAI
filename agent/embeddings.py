import logging
from abc import ABC, abstractmethod
from typing import Optional, List

class BaseEmbeddingProvider(ABC):
    dimension: int = None
    
    def __init__(self):
        """
        Initializes the embedding provider with a fixed dimension.

        Args:
            dimension (int): The fixed dimension of the embeddings.
        """
        if not self.dimension:
            logging.info("Dimension not set, setting dimension from sample")
            self.set_dimension_from_sample()
    
    def set_dimension_from_sample(self):
        """
        Sets the dimension from a sample embedding.
        """
        embedding = self.embed_text("sample text")
        self.dimension = len(embedding)

    @abstractmethod
    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Embeds the given text and returns the embedding vector.

        Args:
            text (str): The text to embed.

        Returns:
            Optional[List[float]]: The embedding vector or None if embedding fails.
        """
        pass