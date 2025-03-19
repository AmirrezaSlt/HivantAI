import logging
from typing import Optional, List, Iterator
import requests
from agent.cognitive_engine.llm import BaseLLMProvider
import json
import time

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
        
        # Add retry logic with exponential backoff
        max_retries = 3
        retry_delay = 1  # starting delay in seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    logging.warning(f"Rate limited by Azure OpenAI (attempt {attempt+1}/{max_retries}). Retrying in {retry_delay} seconds.")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
                else:
                    logging.error(f"API ERROR: {e.response.text}")
                    raise
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                raise

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
        
        # Add retry logic with exponential backoff
        max_retries = 3
        retry_delay = 1  # starting delay in seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=60  # Add explicit timeout
                )
                response.raise_for_status()
                
                # If we reach here, we have a successful response
                break
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    logging.warning(f"Rate limited by Azure OpenAI (attempt {attempt+1}/{max_retries}). Retrying in {retry_delay} seconds.")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
                    continue
                else:
                    logging.error(f"API ERROR: {e.response.text}")
                    # Return an error message in the stream format
                    yield f"Error from Azure OpenAI API: {e.response.status_code} - Rate limit exceeded. Please try again later."
                    return
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                yield f"Unexpected error: {str(e)}"
                return
        
        # If we exhausted all retries, return an error
        if attempt == max_retries - 1:
            yield "Error: Maximum retry attempts reached. Azure OpenAI API is currently unavailable."
            return
            
        try:
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
        except Exception as e:
            logging.error(f"Error during streaming: {str(e)}")
            yield f"Error during streaming: {str(e)}"