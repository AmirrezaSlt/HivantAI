import requests
from typing import Optional, List
from agent.embeddings import BaseEmbeddingProvider

class AzureOpenAIEmbeddingProvider(BaseEmbeddingProvider):
    dimension = 1536

    def __init__(self, 
            api_key: str, 
            endpoint: str, 
            deployment_name: str,
            dimension: int = 1536,
            **kwargs
        ):
        """
        Initializes the Azure OpenAI embedding provider.

        Args:
            api_key (Optional[str]): API key for Azure OpenAI.
            endpoint (Optional[str]): Endpoint URL for Azure OpenAI.
            deployment_name (str): Deployment name for the embedding model.
            dimension (int): Dimension of the embedding vectors.
        """
        super().__init__(dimension=dimension, **kwargs)
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.api_url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/embeddings?api-version=2023-05-15"
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }

    def embed_text(self, text: str) -> Optional[List[float]]:
        try:
            data = {
                "input": text, 
                "model": self.deployment_name, 
                "encoding_format": "float", 
                "dimensions": self.dimension
            }
            response = requests.post(self.api_url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except requests.exceptions.RequestException as e:
            print(f"Embedding request failed: {e}: {response.text}")
            return None
        