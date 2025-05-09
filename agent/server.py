from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
from datetime import datetime
from contextlib import asynccontextmanager
import uvicorn
from agent.logger import logger

class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    attachments: Optional[Dict[str, bytes]] = None
    show_thinking: Optional[bool] = True
    show_tool_requests: Optional[bool] = True
    show_tool_outputs: Optional[bool] = True
    conversation_id: Optional[str] = None

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

class AgentServer:
    def __init__(self, agent):
        """Initialize the server with an agent instance"""
        self.agent = agent
        self.app = None
    
    def setup_app(self):
        """Setup FastAPI application with all routes and middleware"""
        # Context manager for application startup and shutdown
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Agent is already initialized
            logger.info("Agent server starting up")
            yield
            logger.info("Shutting down agent server")

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
            if self.agent is None:
                raise HTTPException(status_code=500, detail="Agent not initialized")
            return self.agent

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: ChatCompletionRequest,
            agent = Depends(get_agent),
            conversation_id: Optional[str] = None
        ):
            # Extract all messages to pass to the agent
            if not request.messages:
                raise HTTPException(status_code=400, detail="No messages provided")
            
            # Get the last user message
            user_messages = [msg for msg in request.messages if msg.role == "user"]
            if not user_messages:
                raise HTTPException(status_code=400, detail="No user message provided")
            
            user_message = user_messages[-1].content
            
            # Use conversation_id from query param if provided, otherwise from request body
            conversation_id = conversation_id or request.conversation_id
            
            # Create input object with attachments if provided
            from agent.input import Input
            input_obj = Input(message=user_message, attachments=request.attachments)
            
            # Convert Pydantic messages to dict format for the agent
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # Handle streaming response
            if request.stream:
                async def stream_response():
                    # Start the response with an empty chunk
                    start_chunk = {
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(start_chunk)}\n\n"
                    
                    # Get the input ready for the agent
                    if agent.retriever:
                        agent.retriever.query_and_retrieve(query=input_obj.message)
                    
                    # Process and stream the response
                    try:
                        # Track content for each tag type to handle closing tags properly
                        tag_content = {}
                        
                        for chunk in agent.respond(input=input_obj, conversation_id=conversation_id):
                            tag = chunk["type"]
                            content = chunk["content"]
                            finished = chunk.get("finished", False)
                            
                            # Skip chunks the user doesn't want to see
                            if (tag == "thinking" and not request.show_thinking or
                                tag == "tool" and not request.show_tool_requests or
                                (tag in ["tool_output", "tool_error"] and not request.show_tool_outputs)):
                                continue
                            
                            # Format the content with appropriate tags
                            formatted_content = ""
                            
                            # Special handling for XML tag formatting
                            if tag == "thinking":
                                # Initialize if first chunk of this tag type
                                if tag not in tag_content:
                                    tag_content[tag] = ""
                                    formatted_content = "<thinking>"
                                
                                # Add the content
                                formatted_content += content
                                tag_content[tag] += content
                                
                                # Close tag if finished
                                if finished:
                                    formatted_content += "</thinking>"
                                    tag_content.pop(tag, None)
                            
                            elif tag in ["tool", "tool_output", "tool_error"]:
                                # These can now be partial responses (for streaming tools)
                                tag_open = f"<{tag}>"
                                tag_close = f"</{tag}>"
                                
                                # For tool_output, handle both streaming and non-streaming cases
                                if tag == "tool_output":
                                    if tag not in tag_content:
                                        # First chunk of this response
                                        tag_content[tag] = content
                                        formatted_content = f"{tag_open}{content}"
                                        
                                        # If it's already finished with the first chunk, close the tag
                                        if finished:
                                            formatted_content += f"{tag_close}"
                                            tag_content.pop(tag, None)
                                    else:
                                        # Continuation of a previous chunk
                                        # Try to parse both the previous and new content as JSON
                                        try:
                                            prev_content = json.loads(tag_content[tag])
                                            new_content = json.loads(content)
                                            
                                            # If they're both JSON objects and have the same keys, update incrementally
                                            if isinstance(prev_content, dict) and isinstance(new_content, dict):
                                                # Find new or modified keys
                                                delta_content = {}
                                                for key, value in new_content.items():
                                                    if key not in prev_content or prev_content[key] != value:
                                                        delta_content[key] = value
                                                
                                                if delta_content:
                                                    # Only send the delta
                                                    formatted_content = json.dumps(delta_content)
                                                    tag_content[tag] = content  # Update the stored content
                                            else:
                                                # For non-object JSON or complete replacement, just send the new content
                                                formatted_content = content
                                                tag_content[tag] = content
                                        except (json.JSONDecodeError, TypeError):
                                            # If either is not valid JSON, treat as text and append
                                            new_text = content[len(tag_content[tag]):] if content.startswith(tag_content[tag]) else content
                                            formatted_content = new_text
                                            tag_content[tag] = content
                                        
                                        # If this chunk has finished, close the tag
                                        if finished:
                                            formatted_content += f"{tag_close}"
                                            tag_content.pop(tag, None)
                                else:
                                    # For tool and tool_error, just wrap with tags
                                    formatted_content = f"{tag_open}{content}{tag_close}"
                            
                            # For regular text, just pass through 
                            elif tag == "answer":
                                formatted_content = content
                            
                            # Skip empty content unless it's a finished tag that needs closing
                            if not formatted_content and not (finished and tag in tag_content):
                                continue
                                
                            # Prepare the delta response chunk
                            delta_chunk = {
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": formatted_content},
                                    "finish_reason": "stop" if finished and tag == "answer" else None
                                }]
                            }
                            
                            # Send the chunk
                            yield f"data: {json.dumps(delta_chunk)}\n\n"
                        
                        # End the stream with a final done message
                        yield "data: [DONE]\n\n"
                    
                    except Exception as e:
                        logger.error(f"Error in stream_response: {str(e)}")
                        error_chunk = {
                            "choices": [{
                                "index": 0,
                                "delta": {"content": f"\n\nError: {str(e)}"},
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                
                return StreamingResponse(
                    stream_response(),
                    media_type="text/event-stream",
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )
            else:
                # Non-streaming response
                try:
                    # Get the input ready for the agent
                    if agent.retriever:
                        agent.retriever.query_and_retrieve(query=input_obj.message)
                    
                    # Get a non-streaming response
                    response_gen = agent.respond(input=input_obj, conversation_id=conversation_id)
                    
                    # For non-streaming, collect all data and return final response
                    response_text = ""
                    for chunk in response_gen:
                        if chunk["type"] == "answer":
                            response_text += chunk["content"]
                    
                    # Construct the response
                    return {
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": response_text
                                },
                                "finish_reason": "stop"
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 0,  # We don't track tokens here
                            "completion_tokens": 0,
                            "total_tokens": 0
                        }
                    }
                except Exception as e:
                    logger.error(f"Error in non-streaming response: {str(e)}")
                    raise HTTPException(status_code=500, detail=str(e))

        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "agent": self.agent.name if self.agent else "None"}
        
        self.app = app
        return app
    
    def start(self, host="0.0.0.0", port=8000):
        """Start the server"""
        if self.app is None:
            self.setup_app()
        
        logger.info(f"Starting agent server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)
