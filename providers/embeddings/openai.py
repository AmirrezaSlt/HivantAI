import os
import requests
from typing import Optional, List
from agent.retriever.embeddings import BaseEmbeddingProvider

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    dimension = 1536

    def __init__(self, api_key: Optional[str], model: str):
        """
        Initializes the OpenAI embedding provider.

        Args:
            api_key (Optional[str]): API key for OpenAI.
            model (str): The model to use for embeddings.
        """
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def embed_text(self, text: str) -> Optional[List[float]]:
        try:
            data = {"model": self.model, "input": text}
            response = requests.post(self.api_url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except requests.exceptions.RequestException as e:
            print(f"Embedding request failed: {e}")
            return None