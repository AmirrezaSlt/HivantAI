import requests
from typing import Optional, List
from agent.embeddings import BaseEmbeddingProvider

class AzureOpenAIEmbeddingProvider(BaseEmbeddingProvider):
    dimension = 1536

    def __init__(self, api_key: Optional[str], endpoint: Optional[str], deployment_name: str):
        """
        Initializes the Azure OpenAI embedding provider.

        Args:
            api_key (Optional[str]): API key for Azure OpenAI.
            endpoint (Optional[str]): Endpoint URL for Azure OpenAI.
            deployment_name (str): Deployment name for the embedding model.
        """
        super().__init__()
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.api_url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/embeddings?api-version=2024-02-01"
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }

    def embed_text(self, text: str) -> Optional[List[float]]:
        try:
            data = {"input": text}
            response = requests.post(self.api_url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except requests.exceptions.RequestException as e:
            print(f"Embedding request failed: {e}")
            return None