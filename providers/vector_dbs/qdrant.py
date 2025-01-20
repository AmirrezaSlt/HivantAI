from qdrant_client import QdrantClient as QClient
from qdrant_client.http import models
from typing import List, Any, Tuple, Dict, Optional
from agent.vector_db.base import BaseVectorDB
import logging
import uuid

class QdrantVectorDB(BaseVectorDB):
    def __init__(self, 
            host: str = "localhost", 
            port: int = 6333, 
            score_threshold: float = 0.0,
            **kwargs
        ):
        super().__init__(score_threshold)
        self.client = QClient(host=host, port=port)
        self.collections = {}

    def setup(self, dimension: int, collection: str = "default") -> None:
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )
        self.collections[collection] = True

    def add_vectors(self, data: List[Tuple[List[float], Dict[str, Any]]], collection: str = "default") -> None:
        points = []
        for vector, metadata in data:
            point_id = str(uuid.uuid4())
            points.append(models.PointStruct(
                id=point_id,
                vector=vector,
                payload=metadata
            ))

        self.client.upsert(
            collection_name=collection,
            points=points
        )

    def query_vectors(
        self, 
        query_vectors: List[List[float]], 
        top_k: int,
        collection: str = "default",
        filter_dict: Optional[Dict] = None
    ) -> List[List[Dict[str, Any]]]:
        results = []
        
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

        for query_vector in query_vectors:
            search_result = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=self.score_threshold,
                query_filter=qdrant_filter
            )
            results.append([
                {
                    'retriever_id': hit.payload['retriever_id'], 
                    'uri': hit.payload['uri'], 
                    'score': hit.score
                } for hit in search_result
            ])
        
        return results

    def teardown(self, collection: str = "default") -> None:
        self.client.delete_collection(collection_name=collection)
        if collection in self.collections:
            del self.collections[collection]

    def delete_vectors(self, filter_dict: Dict, collection: str = "default") -> None:
        must_conditions = []
        for key, value in filter_dict.items():
            must_conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                )
            )
        
        self.client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=must_conditions
                )
            )
        )

    def update_vectors(
        self, 
        retriever_uri_pairs: List[Tuple[str, str, List[float]]], 
        collection: str = "default"
    ) -> None:
        for retriever_id, uri, new_vector in retriever_uri_pairs:
            # Search for the existing point with the given retriever_id and uri
            search_result = self.client.search(
                collection_name=collection,
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
                    collection_name=collection,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector=new_vector,
                            payload={"retriever_id": retriever_id, "uri": uri}
                        )
                    ]
                )
            else:
                self.add_vectors([(new_vector, {"retriever_id": retriever_id, "uri": uri})], collection)
                logging.info(f"Added new vector for retriever_id: {retriever_id} and uri: {uri}")

    def get_stats(self, collection: str = "default") -> Dict:
        try:
            collection_info = self.client.get_collection(collection_name=collection)
            return {
                "count": collection_info.points_count,
                "name": collection,
                "dimension": collection_info.config.params.size,
                "distance": str(collection_info.config.params.distance)
            }
        except Exception as e:
            logging.error(f"Error getting stats for collection {collection}: {str(e)}")
            return {"count": 0, "name": collection}