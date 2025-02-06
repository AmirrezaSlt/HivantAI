from typing import List
from .config import RetrieverConfig
from .reference_documents import BaseReferenceDocument
from agent.logger import logger

class Retriever:
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing Retriever instance.")
        self._config = RetrieverConfig(*args, **kwargs)
        self.embedding_provider = self._config.EMBEDDING_PROVIDER
        self.vector_db = self._config.VECTOR_DB
        self.reference_documents = {}
        for doc in self._config.REFERENCE_DOCUMENTS:
            self.reference_documents[doc.id] = doc
        self.attachments = []
        logger.debug(f"Initialized Retriever with {len(self.reference_documents)} reference documents.")

    def setup(self):
        """Sets up the vector database."""
        logger.debug("Setting up vector database.")
        self.vector_db.setup()
        logger.debug("Vector database setup complete.")

    def load_data_to_vector_db(self):
        """Embeds reference documents and stores the vectors in the vector database."""
        logger.debug("Loading data into the vector database.")
        vectors = []
        for name, document in self.reference_documents.items():
            content = document.get_data() or ""
            embedding = self.embedding_provider.embed_text(content)
            if embedding:
                metadata = document.get_metadata()
                metadata.update({'source': name})
                vectors.append((embedding, metadata))
            else:
                logger.debug(f"Skipping document '{name}' due to missing embedding.")
        if vectors:
            self.vector_db.add_vectors(vectors)
            logger.debug(f"Loaded {len(vectors)} embeddings into the vector database.")
            return len(vectors)
        else:
            logger.error("No embeddings were generated; vector database not updated.")
            return 0

    def query_and_retrieve(self, query: str) -> List[BaseReferenceDocument]:
        """Queries the vector database and retrieves similar reference documents."""
        logger.debug(f"Executing query: {query}")
        query_embedding = self.embedding_provider.embed_text(text=query)
        if not query_embedding:
            logger.error(f"Failed to generate embedding for the query: {query}")
            return []
        results = self.vector_db.find_similar(query_embedding, self._config.NUM_REFERENCE_DOCUMENTS)
        logger.debug(f"Query returned {len(results)} results.")
        reference_documents = [
            self.reference_documents[res['source']]
            for res in results if res['source'] in self.reference_documents
        ]
        return reference_documents