import json
from typing import List, Dict, Any, Optional
from .config import CognitiveEngineConfig
from agent.toolkit import Toolkit
from agent.input import Input
from agent.logger import logger
from .response_parser import ResponseParser
from .prompt_template import PromptTemplate
from .memory import Memory

class CognitiveEngine:
    def __init__(self, *args, **kwargs):
        self.config = CognitiveEngineConfig(*args, **kwargs)
        self.name = self.config.AGENT_NAME
        self.role = self.config.AGENT_ROLE
        self.permissions = self.config.AGENT_PERMISSIONS
        self.provider = self.config.LLM_PROVIDER
        self.response_parser = ResponseParser
        self.prompt_template = PromptTemplate(
            name=self.name,
            role=self.role,
            permissions=self.permissions,
        )
        self.memory = Memory()
        
    def respond(self, 
            input: Input,
            toolkit: Toolkit = None,
            state = None,
        ):
        """
        Core method that processes user input and produces reasoning events.
        
        Args:
            input: The user input message and attachments
            toolkit: Optional toolkit for tool usage
            state: Optional state from previous interactions
            
        Yields:
            Raw reasoning events to be processed by the agent
        """
        self.prompt_template.set_toolkit(toolkit)
        self.memory.conversation.add_message("user", input.message)
        
        # Directly yield events from the reasoning process
        yield from self._reason(self.prompt_template)

    def _reason(self, prompt_template: PromptTemplate):
        """
        Core reasoning method that processes LLM responses and handles different response types.
        
        Args:
            prompt_template: The prompt template with the system message.
            
        Yields:
            Dict objects with different response types (thinking, answer, tool_usage, tool_error, error).
        """
        count = 0
        while count < self.config.MAX_ITERATIONS:
            count += 1
            logger.debug(f"Starting reasoning iteration {count}")
            
            messages = [{
                "role": "system",
                "content": prompt_template.system_prompt,
            }, *self.memory.conversation.messages]
            
            logger.debug(f"Sending system prompt: {prompt_template.system_prompt}...")
            
            parser = self.response_parser()
            
            for event in self._get_response(messages, parser):
                tag = event["type"]
                data = event["content"]
                finished = event["finished"]
                
                logger.debug(f"Parser event: type={tag}, finished={finished}")
                
                if tag == "thinking":
                    yield {"type": "thinking", "content": data, "finished": finished}
                    
                    if finished:
                        self.memory.conversation.add_message("assistant", f"[Thinking] {data}")
                        logger.debug("Thinking phase complete")

                elif tag == "answer":
                    yield {"type": "answer", "content": data, "finished": finished}
                    
                    if finished:
                        self.memory.conversation.add_message("assistant", data)
                        logger.debug("Answer complete, returning response")
                        return
                
                elif tag == "tool" and finished:
                    toolkit = prompt_template.toolkit
                    if toolkit:
                        try:
                            if isinstance(data, dict) and "name" in data and "args" in data:
                                tool_name = data["name"]
                                tool_args = data["args"]
                                try:
                                    # If tool_args is a string that looks like JSON, parse it
                                    if isinstance(tool_args, str):
                                        if tool_args.strip().startswith('{') and tool_args.strip().endswith('}'):
                                            tool_args = json.loads(tool_args)
                                    # If it's already a dict, use it directly
                                except json.JSONDecodeError:
                                    raise ValueError(f"Invalid tool format: {data}")
                                
                                # Prepare the tool request with full JSON content for <tool> tag
                                tool_json = json.dumps({"name": tool_name, "args": tool_args})
                                yield {"type": "tool", "content": tool_json, "finished": True}
                                
                                # Execute the tool
                                output = toolkit.invoke(tool_name, tool_args)
                                if isinstance(output, dict):
                                    output_str = json.dumps(output)
                                else:
                                    output_str = str(output)
                                
                                self.memory.conversation.add_message("system", f"<tool_response>{output_str}</tool_response>")
                                yield {"type": "tool_response", "content": output_str, "finished": True}
                                
                                logger.debug(f"Tool {tool_name} executed successfully")
                            else:
                                error_msg = f"Invalid tool format: {data}"
                                self.memory.conversation.add_message("system", error_msg)
                                yield {"type": "tool_error", "content": error_msg, "finished": True}
                                logger.error(f"Tool execution error: {error_msg}")
                        except Exception as e:
                            tool_name = data["name"] if isinstance(data, dict) and "name" in data else "unknown"
                            error_msg = f"Error using tool {tool_name}: {str(e)}"
                            self.memory.conversation.add_message("system", error_msg)
                            yield {"type": "tool_error", "content": error_msg, "finished": True}
                            logger.error(f"Tool execution error: {error_msg}")
                    else:
                        error_msg = "Tool requested but no toolkit available"
                        self.memory.conversation.add_message("system", error_msg)
                        yield {"type": "tool_error", "content": error_msg, "finished": True}
                        logger.warning("Tool requested with no toolkit available")
        
        # If we reach here, we've hit the maximum iterations
        logger.warning(f"Maximum reasoning iterations ({self.config.MAX_ITERATIONS}) reached without conclusive answer")
        yield {"type": "error", "content": "Maximum reasoning iterations reached without conclusive answer.", "finished": True}

    def _get_response(self, messages: List[dict], parser: ResponseParser):     
        """
        Get a response from the LLM provider and parse it.
        
        Args:
            messages: List of message dictionaries to send to the LLM.
            parser: The ResponseParser instance to use for parsing.
            
        Yields:
            Event dictionaries with type, content, and finished fields.
        """
        if self.provider.supports_streaming:
            logger.debug("Using streaming response")
            for chunk in self.provider.stream_response(
                messages=messages,
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE
            ):
                # Parse the chunk and yield any events
                events = parser.feed(chunk)
                for event in events:
                    yield event
        else:
            logger.debug("Using non-streaming response")
            response = self.provider.generate_response(
                messages=messages,
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE
            )
            # Process the full response
            events = parser.feed(response)
            for event in events:
                yield event