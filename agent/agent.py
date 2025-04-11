from .cognitive_engine import CognitiveEngine
from .retriever import Retriever
from .toolkit import Toolkit
from .input import Input
from typing import Dict, Any, Generator
from agent.memory import Memory

class Agent:
    def __init__(self, toolkit: Toolkit, cognitive_engine: CognitiveEngine, retriever: Retriever):
        self.toolkit = toolkit
        self.cognitive_engine = cognitive_engine
        self.retriever = retriever
    
    def setup(self):
        if self.retriever:
            self.retriever.setup()
    
    def load_data_to_vector_db(self):
        if self.retriever:
            self.retriever.load_data_to_vector_db()
      
    def respond(self, input: Input, conversation_id: str = None):
        """
        Process input and generate a response using the cognitive engine.
        
        Args:
            input: The input message and any attachments
            conversation_id: Optional ID to continue a previous conversation
            
        Returns:
            When not streaming: A tuple containing (response_text, updated_state)
            When streaming: A generator yielding response chunks
        """
        memory = Memory(conversation_id)
        
        # Track the user's message in memory (no persistence)
        memory.add_message("user", input.message)
        
        # Apply buffering to the raw events from the cognitive engine
        for event in self.buffer_events(
            self.cognitive_engine.respond(
                input=input,
                memory=memory,
                toolkit=self.toolkit
            )
        ):
            # If this is an answer chunk, collect it for memory
            if event["type"] == "answer":
                # If this is the final chunk, add to memory (no persistence)
                if event.get("finished", False):
                    # External service would handle persistence 
                    pass
            
            # Yield the event to the caller
            yield event
    
    def start_server(self, host="0.0.0.0", port=8000):
        """
        Start the agent server.
        
        Args:
            host: The host to bind the server to
            port: The port to bind the server to
            
        Returns:
            None
        """
        from .server import AgentServer
        server = AgentServer(self)
        server.start(host=host, port=port)
    
    def buffer_events(self, events_generator: Generator) -> Generator[Dict[str, Any], None, None]:
        """
        Buffers events from the cognitive engine for more efficient streaming.
        
        Args:
            events_generator: Raw event generator from the cognitive engine
            
        Yields:
            Buffered event chunks with delta content
        """
        current_tag = None
        last_content = ""
        
        # Buffers for accumulating tokens of the same tag type
        delta_buffer = ""
        buffer_tag = None
        min_buffer_size = 20  # Minimum number of characters before sending a buffer
        
        for event in events_generator:
            tag = event["type"]
            content = event["content"]
            finished = event.get("finished", False)
            
            # Tool events are always sent immediately
            if tag in ["tool", "tool_output", "tool_error"]:
                # If we have a pending buffer, flush it first
                if delta_buffer:
                    yield {"type": buffer_tag, "content": delta_buffer, "finished": False}
                    delta_buffer = ""
                
                # Send the tool event directly
                yield {"type": tag, "content": content, "finished": finished}
                current_tag = tag
                last_content = content
                continue
            
            # If tag changes, flush any existing buffer
            if tag != buffer_tag and delta_buffer:
                yield {"type": buffer_tag, "content": delta_buffer, "finished": False}
                delta_buffer = ""
            
            # For tag transitions or new tags
            if tag != current_tag:
                # For completely new content with a new tag
                if content:
                    if len(content) < min_buffer_size and not finished:
                        # Start buffering for small content
                        buffer_tag = tag
                        delta_buffer = content
                    else:
                        # Send substantial content directly
                        yield {"type": tag, "content": content, "finished": finished}
                        
                current_tag = tag
                last_content = content
            else:
                # For the same tag, calculate the delta
                if content != last_content:
                    delta = content[len(last_content):]
                    last_content = content
                    
                    if delta:
                        # Add to buffer if of same tag type
                        if tag == buffer_tag:
                            delta_buffer += delta
                            
                            # Flush the buffer if it's large enough or this is the final chunk
                            if len(delta_buffer) >= min_buffer_size or finished:
                                yield {"type": buffer_tag, "content": delta_buffer, "finished": finished}
                                delta_buffer = ""
                                buffer_tag = None if finished else tag
                        else:
                            # Start a new buffer or send directly
                            if len(delta) < min_buffer_size and not finished:
                                buffer_tag = tag
                                delta_buffer = delta
                            else:
                                yield {"type": tag, "content": delta, "finished": finished}
                
                # If content is unchanged but finished status changed, send an empty chunk with finished=True
                elif finished and not event.get("last_finished", False):
                    # Flush any buffer first
                    if delta_buffer:
                        yield {"type": buffer_tag, "content": delta_buffer, "finished": False}
                        delta_buffer = ""
                    
                    yield {"type": tag, "content": "", "finished": True}
            
            # Remember the last finished status for this tag
            event["last_finished"] = finished
            
        # Flush any remaining buffer at the end
        if delta_buffer:
            yield {"type": buffer_tag, "content": delta_buffer, "finished": True}
