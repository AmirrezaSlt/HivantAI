import logging
from typing import List, Dict, Any, Tuple
from .models import CognitiveEventType, CognitiveTrail, ConversationHistory
from ..config import Config
from ..prompt_template import SystemPrompt, ContextPrompt
from ..output_parser import OutputParser, ResponseType
from ..tools import BaseTool

class CognitiveEngine:
    def __init__(self, tools: List[BaseTool], config: Config):
        self.config = config.COGNITIVE_ENGINE
        self.provider = self.config.CHAT_PROVIDER(**self.config.CHAT_PROVIDER_KWARGS)
        self.cognitive_trail = self.config.INITIAL_COGNITIVE_TRAIL
        self.conversation_history = ConversationHistory()
        self.system_prompt = SystemPrompt()
        self.context_prompt = ContextPrompt()
        self.system_message = {"role": "system", "content": self.system_prompt.prompt}
        self.tools = tools
        self.output_parser = OutputParser()

    def respond(self, user_message: str, relevant_documents: List[Dict[str, Any]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        self.conversation_history.add_entry(role="user", content=user_message)
        self.cognitive_trail.add_entry(
            type=CognitiveEventType.RECEIVED_MESSAGE,
            title="Received user message",
            data={"message": user_message}
        )

        if relevant_documents:
            self.cognitive_trail.add_entry(
                type=CognitiveEventType.RELEVANT_DOCUMENTS,
                title="Relevant documents",
                data={"documents": relevant_documents}
            )
        
        response = self._think()
        self.conversation_history.add_entry(role="assistant", content=response)
        return response, self.cognitive_trail

    def _think(self) -> str:
        while True:
            context = self.context_prompt.render(cognitive_trail=self.cognitive_trail, tools=self.tools)
            
            messages = [
                self.system_message,
                *context
            ]
            response, response_type = self._send_message(messages)
            
            if response_type == ResponseType.TOOL_USE:
                tool_name, tool_input = response.tool_name, response.tool_input
                tool_output = self.tools[tool_name].invoke(**tool_input)
                self.cognitive_trail.add_entry(
                    type=CognitiveEventType.USED_TOOL,
                    title=f"Decided to use tool: {tool_name}",
                    data={"tool_name": tool_name, "tool_input": tool_input, "tool_output": tool_output}
                )
            elif response_type == ResponseType.FINAL:
                self.cognitive_trail.add_entry(
                    type=CognitiveEventType.FINAL_RESPONSE,
                    title="Generated final response",
                    data={"response": response}
                )
                return response.content
            elif response_type == ResponseType.CLARIFICATION:
                self.cognitive_trail.add_entry(
                    type=CognitiveEventType.CLARIFICATION_REQUEST,
                    title="Requested clarification",
                    data={"clarification": response}
                )
                return response.question

    def _send_message(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.7) -> Tuple[dict, str]:
        response = self.provider.send_message(messages=messages, max_tokens=max_tokens, temperature=temperature)
        parsed_response = self.output_parser.parse(response)
        return parsed_response, parsed_response.response_type
            
    def reset(self):
        self.conversation_history = ConversationHistory()
        self.cognitive_trail = CognitiveTrail()
        logging.info("Cognitive engine reset")
