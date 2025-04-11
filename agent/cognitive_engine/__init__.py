import json
from typing import List, Dict, Any, Optional
from .config import CognitiveEngineConfig
from agent.toolkit import Toolkit
from agent.input import Input
from agent.logger import logger
from .response_parser import ResponseParser
from .prompt_template import PromptTemplate
from agent.memory import Memory
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
        
    def respond(self, 
            input: Input,
            toolkit: Toolkit = None,
            memory: Memory = None,
        ):
        """
        Core method that processes user input and produces reasoning events.
        
        Args:
            input: The user input message and attachments
            messages: List of previous conversation messages with roles and content
            toolkit: Optional toolkit for tool usage
            
        Yields:
            Raw reasoning events to be processed by the agent
        """
        self.prompt_template.set_toolkit(toolkit)
        
        current_message = {"role": "user", "content": input.message}
        
        all_messages = memory.messages + [current_message]
        
        yield from self._reason(self.prompt_template, all_messages)

    def _reason(self, prompt_template: PromptTemplate, messages: List[Dict[str, str]]):
        """
        Core reasoning method that processes LLM responses and handles different response types.
        
        Args:
            prompt_template: The prompt template with the system message.
            messages: The conversation messages to send to the LLM.
            
        Yields:
            Dict objects with different response types (thinking, answer, tool_usage, tool_error, error).
        """
        count = 0
        while count < self.config.MAX_ITERATIONS:
            count += 1
            logger.debug(f"Starting reasoning iteration {count}")
            
            system_message = {
                "role": "system",
                "content": prompt_template.system_prompt,
            }
            llm_messages = [system_message] + messages
            
            logger.debug(f"Sending system prompt: {prompt_template.system_prompt}...")
            
            parser = self.response_parser()
            
            for event in self._get_response(llm_messages, parser):
                tag = event["type"]
                data = event["content"]
                finished = event["finished"]
                
                logger.debug(f"Parser event: type={tag}, finished={finished}")
                
                if tag == "thinking":
                    yield {"type": "thinking", "content": data, "finished": finished}
                    
                    if finished:
                        # Add thinking message to conversation for next iteration
                        messages.append({"role": "assistant", "content": f"[Thinking] {data}"})
                        logger.debug("Thinking phase complete")

                elif tag == "answer":
                    yield {"type": "answer", "content": data, "finished": finished}
                    
                    if finished:
                        # Return after completed answer
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
                                # Check if tool supports streaming
                                if toolkit.supports_streaming(tool_name):
                                    # Process streaming output
                                    logger.debug(f"Executing tool {tool_name} with streaming")
                                    streaming_count = 0
                                    final_output = None
                                    
                                    for partial_output in toolkit.invoke_stream(tool_name, tool_args):
                                        streaming_count += 1
                                        # Format the partial output
                                        if isinstance(partial_output, dict):
                                            output_str = json.dumps(partial_output)
                                        else:
                                            output_str = str(partial_output)
                                        
                                        # Check if this is a finished chunk
                                        is_finished = False
                                        if isinstance(partial_output, dict) and partial_output.get("finished", False):
                                            is_finished = True
                                            final_output = partial_output
                                        
                                        # Yield the partial output with appropriate finished flag
                                        yield {"type": "tool_output", "content": output_str, "finished": is_finished}
                                        
                                        # If this isn't the final chunk, don't add to conversation memory yet
                                        if not is_finished:
                                            logger.debug(f"Streamed tool output {streaming_count}: partial result")
                                    
                                    # Add the final output to conversation for next iteration
                                    if final_output:
                                        if isinstance(final_output, dict):
                                            final_output_str = json.dumps(final_output)
                                        else:
                                            final_output_str = str(final_output)
                                        
                                        messages.append({"role": "assistant", "content": f"<tool_output>{final_output_str}</tool_output>"})
                                        logger.debug(f"Tool {tool_name} streamed execution completed with {streaming_count} updates")
                                else:
                                    # Non-streaming tool execution
                                    output = toolkit.invoke(tool_name, tool_args)
                                    if isinstance(output, dict):
                                        output_str = json.dumps(output)
                                    else:
                                        output_str = str(output)
                                    
                                    messages.append({"role": "assistant", "content": f"<tool_output>{output_str}</tool_output>"})
                                    yield {"type": "tool_output", "content": output_str, "finished": True}
                                    
                                    logger.debug(f"Tool {tool_name} executed successfully")
                            else:
                                error_msg = f"Invalid tool format: {data}"
                                messages.append({"role": "assistant", "content": error_msg})
                                yield {"type": "tool_error", "content": error_msg, "finished": True}
                                logger.error(f"Tool execution error: {error_msg}")
                        except Exception as e:
                            tool_name = data["name"] if isinstance(data, dict) and "name" in data else "unknown"
                            error_msg = f"Error using tool {tool_name}: {str(e)}"
                            messages.append({"role": "assistant", "content": error_msg})
                            yield {"type": "tool_error", "content": error_msg, "finished": True}
                            logger.error(f"Tool execution error: {error_msg}")
                    else:
                        error_msg = "Tool requested but no toolkit available"
                        messages.append({"role": "assistant", "content": error_msg})
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