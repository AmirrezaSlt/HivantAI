import logging
from typing import List, Dict, Any, Tuple
from .models import ReasoningStepType, ReasoningSteps
from ..config import Config
from .prompt_template import system_prompt_template, context_prompt_template
from ..output_parser import OutputParser, ResponseType
from ..tools import BaseTool
from jinja2 import Template

class CognitiveEngine:
    def __init__(self, config: Config):
        self.config = config.COGNITIVE_ENGINE
        self.provider = self.config.LLM_PROVIDER(**self.config.LLM_PROVIDER_KWARGS)
        self.system_prompt = system_prompt_template
        self.context_prompt = context_prompt_template
        self.reasoning_steps = ReasoningSteps()
        self.output_parser = OutputParser()

    def respond(self, 
            user_message: str, 
            relevant_documents: List[Dict[str, Any]] = None,
            tools: List[BaseTool] = None
        ) -> str:
        self.reasoning_steps.add_entry(
            type=ReasoningStepType.RECEIVED_MESSAGE,
            title="Received user message",
            data={"message": user_message}
        )
        
        response = self._think(relevant_documents, tools)
        return response

    def _think(self, 
            relevant_documents: List[Dict[str, Any]], 
            tools: List[BaseTool]
        ) -> str:

        while True:
            rendered_system_prompt = Template(self.system_prompt).render(
                tools=tools,
            )
            
            rendered_context = Template(self.context_prompt).render(
                reasoning_steps=self.reasoning_steps,
                relevant_documents=relevant_documents,
            )
            
            messages = [
                {"role": "system", "content": rendered_system_prompt},
                {"role": "user", "content": rendered_context}
            ]
            
            response, response_type = self._send_message(messages)
            
            if response_type == ResponseType.TOOL_USE:
                tool_name, tool_input = response.tool_name, response.tool_input
                tool_output = self.tools[tool_name].invoke(**tool_input)
                self.reasoning_steps.add_entry(
                    type=ReasoningStepType.USED_TOOL,
                    title=f"Decided to use tool: {tool_name}",
                    data={"tool_name": tool_name, "tool_input": tool_input, "tool_output": tool_output}
                )
            elif response_type == ResponseType.FINAL:
                self.reasoning_steps.add_entry(
                    type=ReasoningStepType.FINAL_RESPONSE,
                    title="Generated final response",
                    data={"response": response}
                )
                return response.content
            elif response_type == ResponseType.CLARIFICATION:
                self.reasoning_steps.add_entry(
                    type=ReasoningStepType.CLARIFICATION_REQUEST,
                    title="Requested clarification",
                    data={"clarification": response}
                )
                return response.question

    def _send_message(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.7) -> Tuple[dict, str]:
        response = self.provider.generate_response(messages=messages, max_tokens=max_tokens, temperature=temperature)
        parsed_response = self.output_parser.parse(response)
        return parsed_response, parsed_response.response_type
