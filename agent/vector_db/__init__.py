from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional

class BaseVectorDB(ABC):

    @abstractmethod
    def __init__(self, 
        similarity_metric: str, 
        score_threshold: float, 
        dimension: int,
        **kwargs
    ) -> None:
        """
        Initialize the vector database client.
        
        :param similarity_metric: The similarity metric to use for vector comparison
        :param score_threshold: The minimum similarity score for returned results
        :param dimension: The dimension of vectors to be stored
        :raises ValueError: If similarity_metric is not supported by this vector database
        """
        if similarity_metric not in self.supported_similarity_metrics:
            raise ValueError(
                f"Similarity metric {similarity_metric} not supported. "
                f"Supported metrics: {self.supported_similarity_metrics}"
            )
        self.similarity_metric: str = similarity_metric
        self.score_threshold: float = score_threshold
        self.dimension: int = dimension

    @property
    @abstractmethod
    def supported_similarity_metrics(self) -> List[str]:
        """The similarity metrics supported by this vector database."""
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
    def find_similar(
        self, 
        query_vector: List[float], 
        top_k: int,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Find the top_k most similar vectors to the query vector.
        Results are filtered based on the score_threshold and optional filter_dict.
        
        :param query_vector: The query vector to find similar vectors for
        :param top_k: Number of most similar vectors to return
        :param filter_dict: Optional metadata filters (e.g., {"document_type": "pdf"})
        :return: List of results containing 'uri', 'retriever_id', and 'score'
        """
        pass

    @abstractmethod
    def delete_vectors(
        self, 
        filter_dict: Dict
    ) -> None:
        """
        Delete vectors from the vector database based on metadata filters.
        """
        pass

    @abstractmethod
    def update_vectors(
        self, 
        retriever_uri_pairs: List[Tuple[str, str, List[float]]]
    ) -> None:
        """
        Update vectors in the vector database based on the provided retriever-URI-vector pairs.
        If a vector doesn't exist, it should be added as a new entry.
        """
        pass
