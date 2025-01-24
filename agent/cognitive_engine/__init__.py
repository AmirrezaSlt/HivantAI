from typing import List, Dict, Any, Tuple
from pydantic import ValidationError
from .models import ReasoningState, LLMResponse, ReasoningStepType
from .config import CognitiveEngineConfig
from agent.toolkit.tool import BaseTool
from agent.input import Input

class CognitiveEngine:
    def __init__(self, *args, **kwargs):
        self.config = CognitiveEngineConfig(*args, **kwargs)
        self.provider = self.config.LLM_PROVIDER

    def respond(self, 
            input: Input,
            reference_documents: List[Dict[str, Any]] = None,
            tools: List[BaseTool] = None
        ) -> str:

        reasoning_state = ReasoningState(tools=tools)
        reasoning_state.update_state_from_input(input, reference_documents)
        
        response = self._reason(reasoning_state)
        return response

    def _reason(self, reasoning_state: ReasoningState) -> str:
        while True:

            response = self._send_message(reasoning_state.messages)

            if response.step.type == ReasoningStepType.TOOL:
                tool = self.tools[response.step.tool_name]
                result = tool.invoke(**response.step.input_data)
                response.step.output_data = result
                reasoning_state.update_state_from_llm_response(response)
                continue
            
            elif response.step.type == ReasoningStepType.RESPONSE:
                return response.step.response
            
    def _send_message(self, messages: List[dict]) -> Tuple[dict, str]:
        response = self.provider.generate_response(
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        print(response)
        try:
            parsed_response = LLMResponse.model_validate_json(response)
        except ValidationError as e:
            raise ValueError(f"Failed to parse LLM response: {e}")
        return parsed_response
