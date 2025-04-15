import logging
import boto3
from typing import Optional, List, Iterator
from agent.cognitive_engine.llm import BaseLLMProvider
import json
import time

class BedrockLLMProvider(BaseLLMProvider):
    supports_streaming = True

    def __init__(self, 
                 model_id: str = "anthropic.claude-3-7-sonnet-20250219-v1:0", 
                 region_name: str = "us-east-1"):
        super().__init__()
        self.model_id = model_id
        self.region_name = region_name
        
        # Initialize Bedrock client using default credential provider chain
        # (will automatically use credentials from ~/.aws)
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=region_name
        )

    def generate_response(self, messages: List[dict], max_tokens: int, temperature: float) -> Optional[str]:
        # Format messages for Claude
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}]
            })
        
        # Prepare the payload
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "top_k": 250,
            "stop_sequences": [],
            "temperature": temperature,
            "top_p": 0.999,
            "messages": formatted_messages
        }
        
        # Add retry logic with exponential backoff
        max_retries = 3
        retry_delay = 1  # starting delay in seconds
        
        for attempt in range(max_retries):
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(payload)
                )
                
                # Parse response
                response_body = json.loads(response['body'].read().decode('utf-8'))
                return response_body['content'][0]['text']
                
            except self.client.exceptions.ServiceException as e:
                if "ThrottlingException" in str(e) and attempt < max_retries - 1:
                    logging.warning(f"Rate limited by Bedrock (attempt {attempt+1}/{max_retries}). Retrying in {retry_delay} seconds.")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
                else:
                    logging.error(f"API ERROR: {str(e)}")
                    raise
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                raise

    def stream_response(self, messages: List[dict], max_tokens: int, temperature: float) -> Iterator[str]:
        # Format messages for Claude
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}]
            })
        
        # Prepare the payload
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "top_k": 250,
            "stop_sequences": [],
            "temperature": temperature,
            "top_p": 0.999,
            "messages": formatted_messages
        }
        
        # Add retry logic with exponential backoff
        max_retries = 3
        retry_delay = 1  # starting delay in seconds
        
        for attempt in range(max_retries):
            try:
                response = self.client.invoke_model_with_response_stream(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(payload)
                )
                
                # If we reach here, we have a successful response
                break
                
            except self.client.exceptions.ServiceException as e:
                if "ThrottlingException" in str(e) and attempt < max_retries - 1:
                    logging.warning(f"Rate limited by Bedrock (attempt {attempt+1}/{max_retries}). Retrying in {retry_delay} seconds.")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
                    continue
                else:
                    logging.error(f"API ERROR: {str(e)}")
                    yield f"Error from Bedrock API: {str(e)}"
                    return
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                yield f"Unexpected error: {str(e)}"
                return
        
        # If we exhausted all retries, return an error
        if attempt == max_retries - 1:
            yield "Error: Maximum retry attempts reached. Bedrock API is currently unavailable."
            return
            
        try:
            # Process streaming response
            for event in response['body']:
                if 'chunk' in event:
                    chunk_data = json.loads(event['chunk']['bytes'].decode('utf-8'))
                    if content := chunk_data.get('content'):
                        if content and len(content) > 0:
                            yield content[0]['text']
        except Exception as e:
            logging.error(f"Error during streaming: {str(e)}")
            yield f"Error during streaming: {str(e)}" 