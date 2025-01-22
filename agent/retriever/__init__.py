import logging
from typing import List, Dict, Any
from ..config import Config

class Retriever:
    def __init__(self, config: Config):
        self.config = config.RETRIEVER
        self.embedding_provider = self.config.EMBEDDING_PROVIDER
        self.vector_db = self.config.VECTOR_DB
        self.reference_documents = {}
        for doc in self.config.REFERENCE_DOCUMENTS:
            self.reference_documents[doc.id] = doc

    def setup(self):
        """
        Sets up the vector database and embeds the provided content sources.
        """
        self.vector_db.setup()

    def load_data_to_vector_db(self):
        vectors = []
        for reference_document in self.reference_documents:
            content = reference_document.get_data()
            embedding = self.embedding_provider.embed_text(content)
            if embedding:
                metadata = reference_document.get_metadata()
                metadata.update({'source': reference_document.id})
                vectors.append((embedding, metadata))
        if vectors:
            self.vector_db.add_vectors(vectors)
            logging.info(f"Stored embeddings for {len(vectors)} items")
            return len(vectors)
        else:
            logging.error("No vectors to store")
            return 0

    def query_and_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Queries the vector database and retrieves the relevant documents.
        """
        query_embedding = self.embedding_provider.embed_text(text=query)
        if not query_embedding:
            logging.error("Failed to generate embedding for the query.")
            return []

        results = self.vector_db.find_similar(query_embedding, self.config.NUM_RELEVANT_DOCUMENTS)
        relevant_documents = [self.reference_documents[res['source']] for res in results]
        return relevant_documents