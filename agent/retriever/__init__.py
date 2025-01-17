import logging
from typing import List, Dict, Any, Tuple
from ..config import Config
from ..vector_dbs import vector_dbs
from .models import RelevantDocuments, RelevantDocument

class Retriever:
    def __init__(self, config: Config):
        self.config = config.RETRIEVER
        self.embedding_provider = self.config.EMBEDDING_PROVIDER(**self.config.EMBEDDING_PROVIDER_KWARGS)
        # self.vector_db = vector_dbs[self.config.VECTOR_DB_NAME](**self.config.VECTOR_DB_CONFIG)
        self.relevant_documents = RelevantDocuments(documents=[])
        # TODO: Add data sources
        
    def setup(self):
        """
        Sets up the vector database and embeds the provided content sources.
        """
        # self.vector_db.setup(self.embedding_provider.dimension)

    def load_content_to_vector_db(self, retriever_uri_pairs: List[Tuple[str, str]]):
        embedded_contents = []
        for retriever_id, uri in retriever_uri_pairs:
            content = self.content_providers[retriever_id].get_content(uri)
            embedding = self.embedding_provider.embed_text(content)
            if embedding:
                metadata = {'uri': uri, 'retriever_id': retriever_id}
                embedded_contents.append((embedding, metadata))
        self.vector_db.add_vectors(embedded_contents)
        logging.info(f"Stored embeddings for {len(embedded_contents)} items")
        return len(embedded_contents)
    
    def update_relevant_documents(self, retriever_id: str, uri: str, score: float):
        """Add a document to relevant_documents if it doesn't already exist."""
        if not any(
            doc.retriever_id == retriever_id and doc.uri == uri
            for doc in self.relevant_documents.documents
        ):
            content_provider = self.content_providers.get(retriever_id)
            if content_provider:
                link = content_provider.get_resource_link(uri)
                content = content_provider.get_content(uri)
                self.relevant_documents.add_document(RelevantDocument(
                    retriever_id=retriever_id,
                    uri=uri,
                    link=link,
                    score=score,
                    content=content
                ))
            else:
                logging.warning(f"Content provider '{retriever_id}' not found for URI '{uri}'.")

    def query_and_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Queries the vector database and retrieves the relevant documents.
        """
        query_embedding = self.embedding_provider.embed_text(text=query)
        if not query_embedding:
            logging.error("Failed to generate embedding for the query.")
            return []

        results = self.vector_db.query_vectors(query_embedding, self.config.NUM_RELEVANT_DOCUMENTS)
        
        for res in results:
            self.update_relevant_documents(retriever_id=res["retriever_id"], uri=res["uri"], score=res["score"])

        return self.relevant_documents

    def update_embeddings(self, retriever_uri_pairs: List[Tuple[str, str]]):
        """
        Updates the embeddings by re-reading the content sources and updating the vector database.
        """
        embedded_contents = []
        for retriever_id, uri in retriever_uri_pairs:
            content = self.content_providers[retriever_id].get_content(uri)
            embedding = self.embedding_provider.embed_text(content)
            if embedding:
                metadata = {'uri': uri, 'retriever_id': retriever_id}
                embedded_contents.append((embedding, metadata))
        self.vector_db.add_vectors(embedded_contents)
        logging.info(f"Updated embeddings for {len(embedded_contents)} items")
