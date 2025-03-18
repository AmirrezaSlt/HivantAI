import os
import sys
from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Tuple
import json
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager

# Import agent components
from agent.agent import Agent
from agent.input import Input
from agent.cognitive_engine import CognitiveEngine
from agent.retriever import Retriever
from agent.toolkit import Toolkit
from agent.toolkit.config import PythonCodeExecutorConfig

# Import providers
from providers.llm.azure_openai import AzureOpenAILLMProvider
from providers.embeddings.azure_openai import AzureOpenAIEmbeddingProvider
from providers.connections.kubernetes import KubernetesConnection
from providers.vector_dbs.qdrant import QdrantVectorDB
from providers.vector_dbs.qdrant import QdrantVectorDB
from agent.retriever.reference_documents.text_file import TextFileReferenceDocument

# Define request models
class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    state: Optional[Dict[str, Any]] = None
    attachments: Optional[Dict[str, bytes]] = None
    show_thinking: Optional[bool] = True  # Add option to show thinking, default to True
    show_tool_requests: Optional[bool] = True  # Add option to show tool requests, default to True
    show_tool_responses: Optional[bool] = True  # Add option to show tool responses, default to True

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    created: int
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage

class StreamChunk(BaseModel):
    id: str
    created: int
    choices: List[Dict[str, Any]]

# Singleton agent instance
agent_instance = None
agent_state = None

def initialize_agent():
    """Initialize the agent with configurations"""
    
    kubernetes_conn = KubernetesConnection.get_connection()

    cognitive_engine = CognitiveEngine(
        SYSTEM_PROMPT="""
        You are an AI assistant that tries to help the user with their Kubernetes problems. 
        You already have access to the main kubernetes cluster (dg-cluster) by default so assume that's the cluster you're working with and prefer to get data yourself than asking the user for it if possible. 
        You can use the code execution tool to run python code that uses the kubernetes client to run kubernetes commands and get the output.
        Try to do incremental steps and get to a good response and feel free to use the tools multiple times, the previous steps taken will be provided to you. 
        Keep your codes small and atomic and try to debug through multiple steps rather than one large block of code.
        """,
        LLM_PROVIDER=AzureOpenAILLMProvider(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="gpt-4o-default"
        ),
        MAX_ITERATIONS=10,
        AGENT_NAME="Kubernetes Agent",
        AGENT_ROLE="You are an AI assistant that tries to help the user with their Kubernetes problems.",
        AGENT_PERMISSIONS=["kubernetes"]
    )

    retriever = Retriever(
        NUM_REFERENCE_DOCUMENTS=3,
        EMBEDDING_PROVIDER=AzureOpenAIEmbeddingProvider(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="text-embedding-3-small",
            dimension=1536
        ),
        VECTOR_DB=QdrantVectorDB(
            dimension=1536,
            in_memory=True,
            host="vector-db",
            port=6333,
            collection="default",
            score_threshold=0.0,
            similarity_metric="Cosine"
        ),
        # Uncomment and update as needed
        # REFERENCE_DOCUMENTS=[
        #     TextFileReferenceDocument(id="kubectl_debug_guide", file_path="providers/data/kubectl_debug.txt"),
        #     TextFileReferenceDocument(id="contact", file_path="providers/data/contact.txt")
        # ]
    )
    
    toolkit = Toolkit(
        TOOLS=[],
        EXECUTOR=PythonCodeExecutorConfig(
            base_image="python:3.13.1-slim",
            python_version="3.13",
            python_packages=["kubernetes==31.0.0"],
            environment_variables={
                "PYTHONUNBUFFERED": {
                    "value": "1",
                },
                "PYTHONDONTWRITEBYTECODE": {
                    "value": "1",
                }
            },
            resource_requests={"cpu": "200m", "memory": "512Mi"},
            resource_limits={"cpu": "200m", "memory": "512Mi"}
        )
    )

    agent = Agent(
        cognitive_engine=cognitive_engine,
        retriever=retriever,
        toolkit=toolkit
    )
    agent.setup()
    # Uncomment if you want to load vector DB data on startup
    # agent.load_data_to_vector_db()
    
    return agent

# Context manager for application startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize agent on startup
    global agent_instance
    if agent_instance is None:
        agent_instance = initialize_agent()
        print("Agent initialized successfully")
    
    yield
    
    # Cleanup on shutdown if needed
    print("Shutting down the agent server")

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get agent instance
async def get_agent():
    global agent_instance
    if agent_instance is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    return agent_instance

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    agent: Agent = Depends(get_agent)
):
    global agent_state
    
    # Extract the last user message
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")
    
    user_message = user_messages[-1].content
    
    # Create input object with attachments if provided
    input_obj = Input(message=user_message, attachments=request.attachments)
    
    # Use provided state or the global state
    state_to_use = request.state if request.state is not None else agent_state
    
    # Handle streaming response
    if request.stream:
        async def stream_response():
            # Generate a response ID and timestamp
            response_id = f"chatcmpl-{datetime.now().timestamp()}"
            created_time = int(datetime.now().timestamp())
            
            # Start the response with an empty chunk
            start_chunk = {
                "id": response_id,
                "created": created_time,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(start_chunk)}\n\n"
            
            # Content buffer to accumulate the full response
            content_buffer = ""
            new_state = state_to_use
            
            # Stream directly from the cognitive engine
            # Remove the try-except block to let exceptions propagate
            # This will show the full stack trace in the server logs
            
            # Get the input ready for the agent
            reference_documents = agent.retriever.query_and_retrieve(query=input_obj.message) if agent.retriever else None
            
            # Get stream from cognitive engine
            for chunk in agent.cognitive_engine.respond(
                input=input_obj,
                # reference_documents=reference_documents,
                toolkit=agent.toolkit,
                # state=state_to_use
            ):
                # Handle different types of chunks
                if chunk["type"] == "thinking":
                    # Only include thinking chunks if show_thinking is True
                    if request.show_thinking:
                        # Include thinking chunks in the stream with special formatting
                        thinking_content = f"[THINKING] {chunk['content']}"
                        
                        # Add a marker if thinking is finished
                        if chunk.get("finished", False):
                            thinking_content += " [DONE]"
                            
                        sse_chunk = {
                            "id": response_id,
                            "created": created_time,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": thinking_content},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(sse_chunk)}\n\n"
                
                elif chunk["type"] == "tool_usage":
                    # Only include tool request chunks if show_tool_requests is True
                    if request.show_tool_requests:
                        # Format tool request with special formatting
                        tool_request_content = f"[TOOL REQUEST] {chunk['content']}"
                        
                        sse_chunk = {
                            "id": response_id,
                            "created": created_time,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": tool_request_content},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(sse_chunk)}\n\n"
                
                elif chunk["type"] == "tool_error" or chunk["type"] == "tool_response":
                    # Only include tool response chunks if show_tool_responses is True
                    if request.show_tool_responses:
                        # Format tool response with special formatting
                        prefix = "[TOOL ERROR]" if chunk["type"] == "tool_error" else "[TOOL RESPONSE]"
                        tool_response_content = f"{prefix} {chunk['content']}"
                        
                        sse_chunk = {
                            "id": response_id,
                            "created": created_time,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": tool_response_content},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(sse_chunk)}\n\n"
                
                elif chunk["type"] == "answer":
                    # For answer chunks, send them as delta content
                    content = chunk["content"]
                    content_buffer += content
                    
                    # Send the content as a delta
                    sse_chunk = {
                        "id": response_id,
                        "created": created_time,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(sse_chunk)}\n\n"
                
                elif chunk["type"] == "error":
                    # Handle error case
                    error_chunk = {
                        "id": response_id,
                        "created": created_time,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk["content"]},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
            
            # Send final chunk with finish_reason
            final_chunk = {
                "id": response_id,
                "created": created_time,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            
            # Update the agent state with the full response
            agent_state = new_state
            
            # End the response with a final chunk indicating completion
            end_chunk = {
                "id": response_id,
                "created": created_time,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(end_chunk)}\n\n"
            
            # Add an explicit end signal to terminate the connection
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream"
        )
    
    # Handle regular (non-streaming) response
    else:
        # Get response from agent
        response_text, new_state = agent.respond(input=input_obj, state=state_to_use)
        
        # Update global state
        agent_state = new_state
        
        # Create completion response
        completion_response = ChatCompletionResponse(
            id=f"chatcmpl-{datetime.now().timestamp()}",
            created=int(datetime.now().timestamp()),
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionResponseUsage(
                prompt_tokens=len(request.messages),  # This is a simplification, would need actual token counting
                completion_tokens=len(response_text.split()),  # This is a simplification
                total_tokens=len(request.messages) + len(response_text.split())  # This is a simplification
            )
        )
        
        return completion_response

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    
    # Check for required environment variables
    required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
        
    # Start the server
    print("Starting the agent server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
