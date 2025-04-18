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
     "input": {
       "param1": "value1",
       "param2": "value2"
     }
   }
   </tool>
   The tool will be executed immediately after the <tool> tag is seen and the output will be provided back to you.

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
    """A simplified parser for LLM responses that extracts tagged sections."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the parser state."""
        self.buffer = ""
        self.current_tag = None
        self.current_content = ""
    
    def __enter__(self):
        """Context manager entry."""
        self.reset()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            return False
        
        # Process any remaining content
        events = self.finish()
        return True
    
    def finish(self):
        """
        Process any remaining content in the buffer.
        
        Returns:
            List of events generated from remaining content.
        """
        events = []
        
        # If we have a current tag, finalize it
        if self.current_tag:
            event = {"type": self.current_tag, "content": self.current_content, "finished": True}
            events.append(event)
            self.current_tag = None
            self.current_content = ""
        
        # If we have buffer content with no tag, treat as raw
        elif self.buffer.strip():
            event = {"type": "raw", "content": self.buffer.strip(), "finished": True}
            events.append(event)
            self.buffer = ""
        
        return events
    
    def feed(self, chunk):
        """
        Process a chunk of text and extract any tagged content.
        
        Args:
            chunk: String chunk from the LLM response
            
        Returns:
            A list of events, each with {type, content, finished} structure
        """
        # Handle API-specific formatting (OpenAI/Claude)
        if isinstance(chunk, str) and ('data:' in chunk or '{' in chunk):
            try:
                match = re.search(r'data: ({.*})', chunk)
                if match:
                    json_str = match.group(1)
                    data = json.loads(json_str)
                    
                    if 'choices' in data and len(data['choices']) > 0:
                        choice = data['choices'][0]
                        if 'delta' in choice and 'content' in choice['delta']:
                            chunk = choice['delta']['content']
                        elif 'delta' in choice and len(choice['delta']) == 0:
                            return []
            except (json.JSONDecodeError, AttributeError, KeyError) as e:
                logging.debug(f"Not a parseable JSON: {e}")
        
        # Add chunk to buffer
        self.buffer += chunk
        events = []
        
        # Extract complete tags
        for tag_type in ["thinking", "answer", "tool"]:
            pattern = re.compile(f"<{tag_type}>(.*?)</{tag_type}>", re.DOTALL)
            
            # Find all complete instances of this tag
            for match in pattern.finditer(self.buffer):
                content = match.group(1)
                
                if tag_type == "tool":
                    try:
                        # Parse tool content as JSON
                        content = content.strip()
                        # Try to handle both properly formatted JSON and string representation of a dict
                        if content.startswith('{') and content.endswith('}'):
                            tool_data = json.loads(content)
                            # Convert args to a properly formatted JSON string if it's not already
                            if "input" in tool_data and not isinstance(tool_data["input"], str):
                                tool_data["input"] = json.dumps(tool_data["input"])
                        else:
                            # Handle case where content might be a Python dict-like string
                            # Convert Python syntax to JSON syntax
                            content = content.replace("'", "\"")
                            tool_data = json.loads(content)
                            if "input" in tool_data and not isinstance(tool_data["input"], str):
                                tool_data["input"] = json.dumps(tool_data["input"])
                        
                        event = {"type": "tool", "content": tool_data, "finished": True}
                    except json.JSONDecodeError:
                        # If JSON parsing fails, just return the raw content
                        event = {"type": "tool", "content": content, "finished": True}
                else:
                    event = {"type": tag_type, "content": content, "finished": True}
                
                events.append(event)
                
                # Remove the processed tag from the buffer
                start, end = match.span()
                self.buffer = self.buffer[:start] + self.buffer[end:]
                
                # Start over since we modified the buffer
                return events + self.feed("")
        
        # Check for incomplete tags
        if not events:
            for tag_type in ["thinking", "answer", "tool"]:
                start_tag = f"<{tag_type}>"
                end_tag = f"</{tag_type}>"
                
                start_pos = self.buffer.find(start_tag)
                end_pos = self.buffer.find(end_tag)
                
                # If we have a start tag but no end tag, this is an incomplete tag
                if start_pos >= 0 and end_pos < 0:
                    content_start = start_pos + len(start_tag)
                    content = self.buffer[content_start:]
                    
                    # Set current tag and content
                    self.current_tag = tag_type
                    self.current_content = content
                    
                    # Return a partial event
                    return [{"type": tag_type, "content": content, "finished": False}]
                
                # If we have a closing tag without an opening one, ignore it
        
        # If no tags are found but we have content
        if not events and self.buffer.strip() and not self.current_tag:
            event = {"type": "raw", "content": self.buffer, "finished": False}
            return [event]
        
        return events
    
    def get_parsed_response(self):
        """
        Return the current state as a tuple (tag, content, finished).
        
        Returns:
            A tuple (tag, data, finished) for the current state.
        """
        # Check if we have a current tag
        if self.current_tag:
            return self.current_tag, self.current_content, False
        
        # If buffer contains raw content
        if self.buffer.strip():
            return "raw", self.buffer, False
        
        # Default
        return "raw", "", False
