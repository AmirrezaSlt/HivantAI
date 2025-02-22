from typing import Optional, List, Iterator
import requests
from agent.cognitive_engine.llm import BaseLLMProvider
import json

class AzureOpenAILLMProvider(BaseLLMProvider):
    supports_streaming = True

    def __init__(self, api_key: str, endpoint: str, deployment_name: str):
        super().__init__()
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.api_url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version=2024-02-15-preview"
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }

    def generate_response(self, messages: List[dict], max_tokens: int, temperature: float) -> Optional[str]:
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def stream_response(self, messages: List[dict], max_tokens: int, temperature: float) -> Iterator[str]:
        headers = {
            **self.headers,
            "Accept": "text/event-stream"
        }
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            stream=True
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            
            line = line.decode('utf-8')
            if line.startswith('data: '):
                if line.strip() == 'data: [DONE]':
                    break
                    
                try:
                    data = json.loads(line[6:])  # Remove "data: " prefix
                    if choices := data.get('choices', []):
                        if delta := choices[0].get('delta', {}):
                            if content := delta.get('content'):
                                yield content
                except json.JSONDecodeError:
                    continue