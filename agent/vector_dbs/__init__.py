from .qdrant_client import QdrantClient

__all__ = ["QdrantClient"]

vector_dbs = {
    "qdrant": QdrantClient
}