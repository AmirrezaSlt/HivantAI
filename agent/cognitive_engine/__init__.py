from typing import List, Dict, Any, Tuple
from pydantic import ValidationError
from .models import ReasoningState, LLMResponse, ReasoningStepType, ToolUseResponseStep
from .config import CognitiveEngineConfig
from agent.toolkit.tool import BaseTool
from agent.input import Input

class CognitiveEngine:
    def __init__(self, *args, **kwargs):
        self.config = CognitiveEngineConfig(*args, **kwargs)
        self.provider = self.config.LLM_PROVIDER
        self.system_prompt = self.config.SYSTEM_PROMPT or "You are an AI assistant that thinks step by step."
        
    def respond(self, 
            input: Input,
            reference_documents: List[Dict[str, Any]] = None,
            tools: Dict[str, BaseTool] = None,
            state: ReasoningState = None
        ) -> Tuple[str, ReasoningState]:

        reasoning_state = state or ReasoningState(tools=tools, system_prompt=self.system_prompt)
        reasoning_state.update_state_from_input(input, reference_documents)
        
        response = self._reason(reasoning_state)
        return response, reasoning_state

    def _reason(self, reasoning_state: ReasoningState) -> str:
        while len(reasoning_state.trail) < 15:

            response = self._send_message(reasoning_state.messages)
            response_type = response.step.type
            reasoning_state.update_state_from_llm_response(response)

            if response_type == ReasoningStepType.TOOL_USE_REQUEST:
                tool = reasoning_state.tools[response.step.tool_name]
                result = tool.invoke(**response.step.input_data)
                tool_response = ToolUseResponseStep(    
                    output_data=result
                )
                print(tool_response)
                reasoning_state.add_tool_use_response(tool_response)
                continue
                
            elif response_type == ReasoningStepType.CLARIFICATION:
                return response.step.clarification
            
            elif response_type == ReasoningStepType.RESPONSE:
                return response.step.response
            
    def _send_message(self, messages: List[dict]) -> Tuple[dict, str]:
        response = self.provider.generate_response(
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        try:
            parsed_response = LLMResponse.model_validate_json(response)
            print(parsed_response)
        except ValidationError as e:
            print(response)
            raise ValueError(f"Failed to parse LLM response: {e}")
        return parsed_response
