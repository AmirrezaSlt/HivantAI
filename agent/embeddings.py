from abc import ABC, abstractmethod
from typing import Optional, List

class BaseEmbeddingProvider(ABC):
    def __init__(self, dimension: int):
        """
        Initializes the embedding provider.

        Args:
            dimension (int): The size of the embedding vectors this provider produces
        """
        self.dimension = dimension

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