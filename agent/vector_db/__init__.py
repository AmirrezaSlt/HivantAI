from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Dict, Optional

class BaseVectorDB(ABC):
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
    def add_vectors(self, data: List[Tuple[List[float], dict]], collection: str = "default") -> None:
        """Add vectors and metadata to the index."""
        pass

    @abstractmethod
    def query_vectors(
        self, 
        query_vectors: List[List[float]], 
        top_k: int,
        collection: str = "default",
        filter_dict: Optional[Dict] = None
    ) -> List[List[Dict]]:
        """
        Query the index for the top_k most similar vectors.
        Must return a list of lists of dicts with the keys 'uri', 'retriever_id', and 'score'.
        Results should be filtered based on the score_threshold and optional filter_dict.
        
        :param query_vectors: List of query vectors
        :param top_k: Number of results per query
        :param collection: Collection/namespace to query
        :param filter_dict: Optional metadata filters (e.g., {"document_type": "pdf"})
        :return: List of results for each query vector
        """
        pass

    @abstractmethod
    def delete_vectors(
        self, 
        filter_dict: Dict,
        collection: str = "default"
    ) -> None:
        """
        Delete vectors from the vector database based on metadata filters.
        """
        pass

    @abstractmethod
    def update_vectors(
        self, 
        retriever_uri_pairs: List[Tuple[str, str, List[float]]], 
        collection: str = "default"
    ) -> None:
        """
        Update vectors in the vector database based on the provided retriever-URI-vector pairs.
        If a vector doesn't exist, it should be added as a new entry.
        """
        pass

    @abstractmethod
    def get_stats(self, collection: str = "default") -> Dict:
        """
        Get statistics about the collection.
        
        :return: Dictionary containing stats like vector count, dimension, etc.
        """
        pass