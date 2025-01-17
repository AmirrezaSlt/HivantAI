from qdrant_client import QdrantClient as QClient
from qdrant_client.http import models
from typing import List, Any, Tuple, Dict
from .base import BaseVectorDBClient
import logging
import uuid

class QdrantClient(BaseVectorDBClient):
    
    def __init__(self, 
            host: str = "localhost", 
            port: int = 6333, 
            score_threshold: float = 0.0, 
            collection_name: str = "llm"
        ):
        self.client = None
        self.collection_name = collection_name
        self.client = QClient(host=host, port=port)
        self.score_threshold = score_threshold

    def setup(self, dimension: int) -> None:
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )

    def add_vectors(self, data: List[Tuple[List[float], Dict[str, Any]]]) -> int:
        points = []
        for vector, metadata in data:
            point_id = str(uuid.uuid4())
            points.append(models.PointStruct(
                id=point_id,
                vector=vector,
                payload=metadata
            ))

        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return operation_info

    def query_vectors(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=self.score_threshold
        )
        return [{'retriever_id': hit.payload['retriever_id'], 'uri': hit.payload['uri'], 'score': hit.score} for hit in search_result]

    def teardown(self) -> None:
        self.client.delete_collection(collection_name=self.collection_name)

    def delete_vectors(self, ids: List[int]) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=ids
            )
        )

    def update_vectors(self, retriever_uri_pairs: List[Tuple[str, str, List[float]]]) -> None:
        for retriever_id, uri, new_vector in retriever_uri_pairs:
            # Search for the existing point with the given retriever_id and uri
            search_result = self.client.search(
                collection_name=self.collection_name,
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
                # Update the existing point's vector and payload
                self.client.update_points(
                    collection_name=self.collection_name,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector=new_vector,
                            payload={"retriever_id": retriever_id, "uri": uri}
                        )
                    ]
                )
            else:
                # If the point doesn't exist, add it as a new point
                self.add_vectors([(new_vector, {"retriever_id": retriever_id, "uri": uri})])
                logging.info(f"Added new vector for retriever_id: {retriever_id} and uri: {uri}")