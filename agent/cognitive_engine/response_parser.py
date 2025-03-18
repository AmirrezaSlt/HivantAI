import json
import re
import logging

RESPONSE_FORMAT_PROMPT = """
Your response MUST be formatted with specific tags for proper processing:

1. THINKING PHASE: Always start with a thinking phase to reason through the problem.
   <thinking>
   [Your detailed reasoning about the problem goes here]
   </thinking>

2. TOOL USAGE (if needed): If you need to use a tool, specify it clearly.
   <tool>
   {
     "name": "tool_name",
     "args": {
       "param1": "value1",
       "param2": "value2"
     }
   }
   </tool>

3. FINAL ANSWER: Always end with a clear answer.
   <answer>
   [Your final response to the user goes here]
   </answer>

IMPORTANT RULES:
- ALWAYS wrap your entire response in one of these tags
- NEVER skip the <thinking> phase
- Your final response must ALWAYS be in the <answer> tag
- Format all JSON correctly within the <tool> tag
- The system will execute tools as soon as they're requested

Here's an example of good formatting:
<thinking>
I need to solve this math problem. Let me break it down step by step...
</thinking>
<answer>
The solution is 42.
</answer>
"""

class ResponseParser:
    """Parses streaming content to extract tagged sections."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the parser state to start fresh."""
        self.buffer = ""
        self.current_tag = None
        self.current_content = ""
        self.events_queue = []  # Queue to store processed events
        self.open_tags = {}  # Track open tags and their start positions
    
    def __enter__(self):
        """Context manager entry."""
        self.reset()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit, ensuring all tags are closed.
        
        If there's an exception during processing, it will be propagated.
        """
        if exc_type is not None:
            # Don't swallow exceptions
            return False
        
        # Complete processing
        try:
            self.finish()
        except Exception as e:
            # Log the error but don't crash the app
            logging.error(f"Error during parser finish: {e}")
            
        return True
    
    def finish(self):
        """
        Ensure all tags are properly closed and processing is complete.
        Handle any remaining content in the buffer.
        """
        # First, try to process any complete tags that might be in the buffer
        events = self.feed("")
        
        # If we have unprocessed content in the buffer, handle it
        if self.buffer.strip():
            logging.debug(f"Processing remaining buffer content at finish: {self.buffer[:50]}...")
            
            # Check if we have any complete tags
            has_complete_tags = any(event["finished"] for event in self.events_queue)
            
            # Check for content that contains both tags
            if "</thinking>" in self.buffer and "<answer>" in self.buffer:
                # Extract just the answer part
                parts = self.buffer.split("<answer>")
                if len(parts) > 1:
                    answer_text = parts[1]
                    # Remove closing tag if present
                    if "</answer>" in answer_text:
                        answer_text = answer_text.split("</answer>")[0]
                    
                    logging.debug(f"Extracted answer from buffer with split: {answer_text[:50]}...")
                    event = {"type": "answer", "content": answer_text, "finished": True}
                    self.events_queue.append(event)
                    self.buffer = ""
                    return
            
            # If no complete tags found, treat buffer as an answer
            if not has_complete_tags:
                event = {"type": "answer", "content": self.buffer.strip(), "finished": True}
                self.events_queue.append(event)
            # Otherwise add as raw text
            else:
                event = {"type": "raw", "content": self.buffer.strip(), "finished": True}
                self.events_queue.append(event)
            
            # Clear buffer
            self.buffer = ""
    
    def feed(self, chunk):
        """
        Feed a chunk of text to the parser and extract complete tags.
        
        Args:
            chunk: String chunk from the LLM stream
            
        Returns:
            List of events, each with {type, content, finished} structure
        """
        # Handle JSON format responses from OpenAI/Claude API
        if isinstance(chunk, str) and ('data:' in chunk or '{' in chunk):
            try:
                # For streaming responses in the SSE format "data: {...}"
                match = re.search(r'data: ({.*})', chunk)
                if match:
                    json_str = match.group(1)
                    data = json.loads(json_str)
                    
                    # Handle OpenAI/Claude format
                    if 'choices' in data and len(data['choices']) > 0:
                        choice = data['choices'][0]
                        if 'delta' in choice and 'content' in choice['delta']:
                            # Extract just the content
                            content = choice['delta']['content']
                            logging.debug(f"Extracted content from API response: {content}")
                            chunk = content
                        elif 'delta' in choice and len(choice['delta']) == 0:
                            # Empty delta means end of response
                            logging.debug("End of response detected")
                            return []
            except (json.JSONDecodeError, AttributeError, KeyError) as e:
                logging.debug(f"Not a parseable JSON: {e}")
                # Continue with original chunk if not a valid JSON
                pass
        
        # Add chunk to our buffer
        self.buffer += chunk
        new_events = []
        
        # Track open and close tags for better handling across chunks
        for tag_type in ["thinking", "answer", "tool"]:
            open_tag = f"<{tag_type}>"
            close_tag = f"</{tag_type}>"
            
            # Check for opening tags if not already tracking this tag
            if tag_type not in self.open_tags:
                open_pos = self.buffer.find(open_tag)
                if open_pos >= 0:
                    self.open_tags[tag_type] = {
                        "start_pos": open_pos,
                        "content_start": open_pos + len(open_tag)
                    }
                    logging.debug(f"Found opening {tag_type} tag at position {open_pos}")
            
            # Check for closing tags if we're tracking an open tag
            if tag_type in self.open_tags:
                close_pos = self.buffer.find(close_tag)
                if close_pos >= 0:
                    start_info = self.open_tags[tag_type]
                    content = self.buffer[start_info["content_start"]:close_pos]
                    
                    # Create a complete event for this tag
                    if tag_type == "tool":
                        # Parse tool content as JSON
                        try:
                            tool_data = json.loads(content.strip())
                            event = {"type": "tool", "content": tool_data, "finished": True}
                        except json.JSONDecodeError as e:
                            logging.warning(f"Failed to parse tool content as JSON: {e}")
                            event = {"type": "tool", "content": content, "finished": True}
                    else:
                        event = {"type": tag_type, "content": content, "finished": True}
                        
                    new_events.append(event)
                    self.events_queue.append(event)
                    
                    # Remove this tag from tracking and update buffer to remove processed content
                    del self.open_tags[tag_type]
                    self.buffer = self.buffer[close_pos + len(close_tag):]
                    logging.debug(f"Found complete {tag_type} tag: {content[:50]}...")
                    
                    # Start over with processing since buffer is modified
                    return new_events + self.feed("")
        
        # Check if we have a complete <thinking> tag using regex (fallback)
        thinking_pattern = re.compile(r'<thinking>(.*?)</thinking>', re.DOTALL)
        thinking_match = thinking_pattern.search(self.buffer)
        
        if thinking_match:
            # Found a complete thinking tag
            thinking_content = thinking_match.group(1)
            event = {"type": "thinking", "content": thinking_content, "finished": True}
            new_events.append(event)
            self.events_queue.append(event)
            
            # Remove the processed thinking tag from the buffer
            self.buffer = self.buffer[thinking_match.end():]
            logging.debug(f"Extracted complete thinking tag: {thinking_content[:50]}...")
        
        # Check if we have a complete <answer> tag
        answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        answer_match = answer_pattern.search(self.buffer)
        
        if answer_match:
            # Found a complete answer tag
            answer_content = answer_match.group(1)
            event = {"type": "answer", "content": answer_content, "finished": True}
            new_events.append(event)
            self.events_queue.append(event)
            
            # Remove the processed answer tag from the buffer
            self.buffer = self.buffer[answer_match.end():]
            logging.debug(f"Extracted complete answer tag: {answer_content[:50]}...")
        
        # Check if we have a complete <tool> tag
        tool_pattern = re.compile(r'<tool>(.*?)</tool>', re.DOTALL)
        tool_match = tool_pattern.search(self.buffer)
        
        if tool_match:
            # Found a complete tool tag
            tool_content = tool_match.group(1)
            
            # Try to parse the tool content as JSON
            try:
                tool_data = json.loads(tool_content.strip())
                event = {"type": "tool", "content": tool_data, "finished": True}
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse tool content as JSON: {e}")
                event = {"type": "tool", "content": tool_content, "finished": True}
                
            new_events.append(event)
            self.events_queue.append(event)
            
            # Remove the processed tool tag from the buffer
            self.buffer = self.buffer[tool_match.end():]
            logging.debug(f"Extracted complete tool tag: {tool_content[:50]}...")
        
        # If no complete tags found, check for partial tags that weren't previously identified
        if not new_events and not self.open_tags:
            # Check for opening tags without closing tags
            for tag_type in ["thinking", "answer", "tool"]:
                open_tag = f"<{tag_type}>"
                close_tag = f"</{tag_type}>"
                
                open_pos = self.buffer.find(open_tag)
                if open_pos >= 0:
                    # Skip if we also have a closing tag (which will be handled in the next iteration)
                    close_pos = self.buffer.find(close_tag)
                    if close_pos >= 0:
                        continue
                        
                    # Found an opening tag without a closing tag
                    content = self.buffer[open_pos + len(open_tag):]
                    event = {"type": tag_type, "content": content, "finished": False}
                    new_events.append(event)
                    self.events_queue.append(event)
                    
                    # Add to open tags tracking
                    self.open_tags[tag_type] = {
                        "start_pos": open_pos,
                        "content_start": open_pos + len(open_tag)
                    }
                    
                    # Clear buffer since we've processed everything
                    self.buffer = ""
                    logging.debug(f"Found incomplete {tag_type} tag: {content[:50]}...")
                    break
        
        # If no tags found at all, add the buffer as raw content
        if not new_events and self.buffer.strip() and not self.open_tags:
            event = {"type": "raw", "content": self.buffer, "finished": False}
            new_events.append(event)
            self.events_queue.append(event)
            
            # Don't clear buffer, we'll keep accumulating until we find tags
            logging.debug(f"No tags found, keeping as raw: {self.buffer[:50]}...")
        
        return new_events
    
    def get_queue(self):
        """Return the current queue of events."""
        return self.events_queue
    
    def clear_queue(self):
        """Clear the events queue."""
        self.events_queue = []
        
    def get_parsed_response(self):
        """
        Get the parsed response from the events queue.
        
        Returns:
            A tuple (tag, data, finished) where:
            - tag is the type of response (thinking, tool, answer, raw)
            - data is the content of the response
            - finished is a boolean indicating if the response is complete
        """
        # Print the complete contents of the queue for debugging
        print(f"QUEUE-DEBUG: Current events_queue: {self.events_queue}")
        
        # First, check for cases where tags are embedded in the text
        for event in self.events_queue:
            content = event["content"]
            if isinstance(content, str) and "</thinking>" in content and "<answer>" in content:
                # Extract just the answer part
                parts = content.split("<answer>")
                if len(parts) > 1:
                    answer_text = parts[1]
                    # Remove closing tag if present
                    if "</answer>" in answer_text:
                        answer_text = answer_text.split("</answer>")[0]
                    
                    logging.debug(f"Extracted embedded answer: {answer_text[:50]}...")
                    return "answer", answer_text, True
        
        # Modified tag priority processing to ensure thinking chunks are visible
        # First, check for thinking chunks - they should take precedence during reasoning
        for event in self.events_queue:
            if event["type"] == "thinking":
                print(f"PARSER-RETURN-THINKING: Found thinking chunk: {event}")
                logging.debug(f"Returning thinking: {event['content'][:50]}...")
                # Clear it from the queue so we don't return it again
                self.events_queue.remove(event)
                return "thinking", event["content"], event["finished"]
                
        # If no thinking chunks, try to get a finished answer
        for event in self.events_queue:
            if event["type"] == "answer" and event["finished"]:
                logging.debug(f"Found finished answer: {event['content'][:50]}...")
                return "answer", event["content"], True
                
        # Next, try to get a finished tool call
        for event in self.events_queue:
            if event["type"] == "tool" and event["finished"]:
                logging.debug(f"Found finished tool call: {event['content']}")
                return "tool", event["content"], True
                
        # If we already checked for thinking above, no need to do it again
        
        # If no finished events, check for any unfinished events in priority order
        for type_priority in ["answer", "tool"]:
            for event in self.events_queue:
                if event["type"] == type_priority and not event["finished"]:
                    logging.debug(f"Found unfinished {type_priority}: {event['content'][:50]}...")
                    return event["type"], event["content"], False
        
        # If still no events found, check for any raw content
        for event in self.events_queue:
            if event["type"] == "raw":
                logging.debug(f"Found raw content: {event['content'][:50]}...")
                return "raw", event["content"], event["finished"]
        
        # If nothing found, return empty raw
        logging.debug("No events found in queue")
        return "raw", "", False
