from abc import ABC, abstractmethod
from typing import List, Any, Tuple

class BaseVectorDBClient(ABC):
    @abstractmethod
    def __init__(self, score_threshold: float = 0.5, **kwargs) -> None:
        """
        Initialize the vector database client.
        
        :param score_threshold: The minimum similarity score for returned results.
        """
        pass

    @abstractmethod
    def setup(self, **kwargs) -> None:
        """Set up the tables, create indexes, etc."""
        pass

    @abstractmethod
    def teardown(self, **kwargs) -> None:
        """Tear down the vector database."""
        pass

    @abstractmethod
    def add_vectors(self, data: List[Tuple[List[float], dict]]) -> None:
        """Add vectors and metadata to the index."""
        pass

    @abstractmethod
    def query_vectors(self, query_vector: List[float], top_k: int) -> List[Any]:
        """
        Query the index for the top_k most similar vectors.
        Must return a list of dicts with the keys 'uri', 'retriever_id', and 'score'.
        Results should be filtered based on the score_threshold.
        """
        pass

    @abstractmethod
    def delete_vectors(self, query_vector: List[float]) -> None:
        """Delete vectors from the vector database."""
        pass

    @abstractmethod
    def update_vectors(self, retriever_uri_pairs: List[Tuple[str, str, List[float]]]) -> None:
        """
        Update vectors in the vector database based on the provided retriever-URI-vector pairs.
        If a vector doesn't exist, it should be added as a new entry.
        """
        pass