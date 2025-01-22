from qdrant_client import QdrantClient as QClient
from qdrant_client.http import models
from typing import List, Any, Tuple, Dict, Optional
from agent.vector_db import BaseVectorDB
import logging
import uuid

class QdrantVectorDB(BaseVectorDB):
    supported_similarity_metrics = ["Cosine", "Euclid", "Dot", "Manhattan"]

    def __init__(self, 
            dimension: int,
            host: str = "localhost", 
            port: int = 6333, 
            score_threshold: float = 0.0,
            collection: str = "default",
            similarity_metric: str = "Cosine",
            **kwargs
        ):
        super().__init__(
            similarity_metric=similarity_metric,
            score_threshold=score_threshold,
            dimension=dimension,
            **kwargs
        )
        self.client = QClient(host=host, port=port)
        self.collection = collection

    def setup(self) -> None:
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=self.dimension, distance=self.similarity_metric),
        )

    def teardown(self) -> None:
        self.client.delete_collection(collection_name=self.collection)

    def add_vectors(self, data: List[Tuple[List[float], Dict[str, Any]]]) -> None:
        points = []
        for vector, metadata in data:
            point_id = str(uuid.uuid4())
            points.append(models.PointStruct(
                id=point_id,
                vector=vector,
                payload=metadata
            ))

        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

    def find_similar(
        self, 
        query_vector: List[float], 
        top_k: int,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        # Convert filter_dict to Qdrant filter format if provided
        qdrant_filter = None
        if filter_dict:
            must_conditions = []
            for key, value in filter_dict.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            qdrant_filter = models.Filter(must=must_conditions)

        search_result = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=self.score_threshold,
            query_filter=qdrant_filter
        )
        
        return [{
            'source': hit.payload['source'],
            'metadata': hit.payload,
            'score': hit.score
        } for hit in search_result]

    def delete_vectors(self, filter_dict: Dict) -> None:
        must_conditions = []
        for key, value in filter_dict.items():
            must_conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                )
            )
        
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=must_conditions
                )
            )
        )

    def update_vectors(
        self, 
        retriever_uri_pairs: List[Tuple[str, str, List[float]]]
    ) -> None:
        for retriever_id, uri, new_vector in retriever_uri_pairs:
            # Search for the existing point with the given retriever_id and uri
            search_result = self.client.search(
                collection_name=self.collection,
                query_vector=[0] * len(new_vector),  # Dummy vector for metadata search
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="retriever_id",
                            match=models.MatchValue(value=retriever_id)
                        ),
                        models.FieldCondition(
                            key="uri",
                            match=models.MatchValue(value=uri)
                        )
                    ]
                ),
                limit=1
            )

            if search_result:
                point_id = search_result[0].id
                self.client.update_points(
                    collection_name=self.collection,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector=new_vector,
                            payload={"retriever_id": retriever_id, "uri": uri}
                        )
                    ]
                )
            else:
                self.add_vectors([(new_vector, {"retriever_id": retriever_id, "uri": uri})])
                logging.info(f"Added new vector for retriever_id: {retriever_id} and uri: {uri}")
