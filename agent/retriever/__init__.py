import logging
from typing import List, Dict, Any, Tuple
from ..config import Config
from .models import RelevantDocuments, RelevantDocument

class Retriever:
    def __init__(self, config: Config):
        self.config = config.RETRIEVER
        self.embedding_provider = self.config.EMBEDDING_PROVIDER(**self.config.EMBEDDING_PROVIDER_KWARGS)
        self.vector_db = self.config.VECTOR_DB(**self.config.VECTOR_DB_KWARGS)
        self.data_sources = self.config.DATA_SOURCES

    def setup(self):
        """
        Sets up the vector database and embeds the provided content sources.
        """
        self.vector_db.setup(self.embedding_provider.dimension)

    def load_data_to_vector_db(self):
        vectors = []
        for name, data_source in self.data_sources.items():
            content = data_source.get_content()
            embedding = self.embedding_provider.embed_text(content)
            if embedding:
                metadata = {'source': name}
                vectors.append((embedding, metadata))
        self.vector_db.add_vectors(vectors)
        logging.info(f"Stored embeddings for {len(vectors)} items")
        return len(vectors)

    def query_and_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Queries the vector database and retrieves the relevant documents.
        """
        query_embedding = self.embedding_provider.embed_text(text=query)
        if not query_embedding:
            logging.error("Failed to generate embedding for the query.")
            return []

        results = self.vector_db.query_vectors(query_embedding, self.config.NUM_RELEVANT_DOCUMENTS)
        data_sources = [self.data_sources[res['source']] for res in results]
        return data_sources