import json
from typing import List
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
        ):

        self.prompt_template.set_toolkit(toolkit)
        self.memory.conversation.add_message("user", input.message)
        
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
            
            with self.response_parser() as parser:
                # Construct messages for LLM
                messages = [{
                    "role": "system",
                    "content": prompt_template.system_prompt,
                }, *self.memory.conversation.messages]
                
                logger.debug(f"Sending system prompt: {prompt_template.system_prompt[:200]}...")
                
                # Process the LLM response
                self._get_response(messages, parser)
                
                # Get tag, data, and finished flag from parser
                tag, data, finished = parser.get_parsed_response()
                logger.debug(f"Parser response: tag={tag}, finished={finished}, data={data[:100] if isinstance(data, str) else data}")
                
                # Handle different types of chunks
                if tag == "thinking":
                    # Stream thinking for real-time feedback
                    yield {"type": "thinking", "content": data, "finished": finished}
                    
                    # If the thinking is finished, add to memory and continue to next iteration
                    if finished:
                        self.memory.conversation.add_message("assistant", f"[Thinking] {data}")
                        logger.debug("Thinking phase complete, continuing to next iteration")
                        # Continue to next iteration to get final answer
                        continue
                
                elif tag == "answer":
                    # Stream answer content
                    yield {"type": "answer", "content": data, "finished": finished}
                    
                    # If the answer is finished, add to memory and end generation
                    if finished:
                        self.memory.conversation.add_message("assistant", data)
                        logger.debug("Answer complete, returning response")
                        return
                
                elif tag == "tool" and finished:
                    # Process tool usage
                    toolkit = prompt_template.toolkit
                    if toolkit:
                        try:
                            # Better handling of tool data which could be a string or a dict
                            if isinstance(data, dict) and "name" in data and "args" in data:
                                # Valid tool data structure
                                tool_name = data["name"]
                                tool_args = data["args"]
                                if isinstance(tool_args, str):
                                    # Try to parse string args as JSON
                                    try:
                                        tool_args = json.loads(tool_args)
                                    except json.JSONDecodeError:
                                        # If parsing fails, use as-is
                                        pass
                                
                                output = toolkit.invoke(tool_name, tool_args)
                                # Convert output to string if it's a dictionary
                                if isinstance(output, dict):
                                    output_str = json.dumps(output)
                                else:
                                    output_str = str(output)
                                self.memory.conversation.add_message("system", output_str)
                                
                                # First yield a tool request chunk
                                tool_request_msg = f"Using tool: {tool_name}"
                                if isinstance(tool_args, dict):
                                    tool_request_msg += f" with args: {json.dumps(tool_args)}"
                                yield {"type": "tool_usage", "content": tool_request_msg}
                                
                                # Then yield a tool response chunk with the output
                                yield {"type": "tool_response", "content": output_str}
                                logger.debug(f"Tool {tool_name} executed successfully")
                            else:
                                # Invalid tool data structure
                                error_msg = f"Invalid tool format: {data}"
                                self.memory.conversation.add_message("system", error_msg)
                                yield {"type": "tool_error", "content": error_msg}
                                logger.error(f"Tool execution error: {error_msg}")
                        except Exception as e:
                            # Safe error reporting that doesn't assume data structure
                            tool_name = data["name"] if isinstance(data, dict) and "name" in data else "unknown"
                            error_msg = f"Error using tool {tool_name}: {str(e)}"
                            self.memory.conversation.add_message("system", error_msg)
                            yield {"type": "tool_error", "content": error_msg}
                            logger.error(f"Tool execution error: {error_msg}")
                    else:
                        error_msg = "Tool requested but no toolkit available"
                        self.memory.conversation.add_message("system", error_msg)
                        yield {"type": "tool_error", "content": error_msg}
                        logger.warning("Tool requested with no toolkit available")
                
                # If we have a raw response or unfinished content, continue to next iteration
                elif tag == "raw" or not finished:
                    logger.debug(f"No conclusive result in iteration {count}, continuing...")
                    continue
        
        # If we reach here, we've hit the maximum iterations
        logger.warning(f"Maximum reasoning iterations ({self.config.MAX_ITERATIONS}) reached without conclusive answer")
        yield {"type": "error", "content": "Maximum reasoning iterations reached without conclusive answer."}

    def _get_response(self, messages: List[dict], parser: ResponseParser):     
        if self.provider.supports_streaming:
            logger.debug("Using streaming response")
            for chunk in self.provider.stream_response(
                messages=messages,
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE
            ):
                events = parser.feed(chunk)
        else:
            logger.debug("Using non-streaming response")
            response = self.provider.generate_response(
                messages=messages,
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE
            )
            events = parser.feed(response)