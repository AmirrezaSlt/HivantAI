from typing import List, Dict, Tuple, Optional
import chromadb
from chromadb.config import Settings
from . import BaseVectorDB

class ChromaVectorDB(BaseVectorDB):
    def __init__(self, score_threshold: float = 0.5, persist_directory: str = None):
        super().__init__(score_threshold)
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        self.collections = {}

    def setup(self, **kwargs) -> None:
        pass

    def teardown(self, **kwargs) -> None:
        self.client.reset()

    def add_vectors(self, data: List[Tuple[List[float], dict]], collection: str = "default") -> None:
        if collection not in self.collections:
            self.collections[collection] = self.client.create_collection(name=collection)
        
        coll = self.collections[collection]
        embeddings = []
        metadatas = []
        ids = []
        
        for vector, metadata in data:
            embeddings.append(vector)
            metadatas.append(metadata)
            ids.append(f"{metadata['retriever_id']}_{metadata['uri']}")
        
        coll.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def query_vectors(
        self, 
        query_vectors: List[List[float]], 
        top_k: int,
        collection: str = "default",
        filter_dict: Optional[Dict] = None
    ) -> List[List[Dict]]:
        if collection not in self.collections:
            return [[] for _ in query_vectors]
        
        coll = self.collections[collection]
        results = []
        
        for query_vector in query_vectors:
            query_result = coll.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=filter_dict
            )
            
            formatted_results = []
            for i in range(len(query_result['ids'][0])):
                score = query_result['distances'][0][i]
                if score <= self.score_threshold:
                    metadata = query_result['metadatas'][0][i]
                    formatted_results.append({
                        'uri': metadata['uri'],
                        'retriever_id': metadata['retriever_id'],
                        'score': score
                    })
            results.append(formatted_results)
        
        return results

    def delete_vectors(self, filter_dict: Dict, collection: str = "default") -> None:
        if collection in self.collections:
            coll = self.collections[collection]
            coll.delete(where=filter_dict)

    def update_vectors(
        self, 
        retriever_uri_pairs: List[Tuple[str, str, List[float]]], 
        collection: str = "default"
    ) -> None:
        if collection not in self.collections:
            self.collections[collection] = self.client.create_collection(name=collection)
        
        coll = self.collections[collection]
        
        for retriever_id, uri, vector in retriever_uri_pairs:
            id = f"{retriever_id}_{uri}"
            coll.upsert(
                ids=[id],
                embeddings=[vector],
                metadatas=[{'retriever_id': retriever_id, 'uri': uri}]
            )

    def get_stats(self, collection: str = "default") -> Dict:
        if collection not in self.collections:
            return {"count": 0}
        
        coll = self.collections[collection]
        return {
            "count": coll.count(),
            "name": collection
        } 