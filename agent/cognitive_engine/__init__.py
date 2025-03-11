from typing import List, Dict, Any, Tuple
from pydantic import ValidationError
from .models import ReasoningState, LLMResponse, ReasoningStepType, ToolUseResponseStep
from .config import CognitiveEngineConfig
from agent.toolkit import Toolkit
from agent.input import Input
from agent.logger import logger

class CognitiveEngine:
    def __init__(self, *args, **kwargs):
        self.config = CognitiveEngineConfig(*args, **kwargs)
        self.provider = self.config.LLM_PROVIDER
        self.system_prompt = self.config.SYSTEM_PROMPT or "You are an AI assistant."
        logger.debug(f"CognitiveEngine initialized with system_prompt: {self.system_prompt}")
        
    def respond(self, 
            input: Input,
            reference_documents: List[Dict[str, Any]] = None,
            toolkit: Toolkit = None,
            state: ReasoningState = None
        ) -> Tuple[str, ReasoningState]:
        logger.debug(f"Input received: {input}")
        if reference_documents:
            logger.debug(f"Reference documents provided: {reference_documents}")
        if toolkit:
            logger.debug(f"Toolkit provided: {toolkit}")
            
        reasoning_state = state or ReasoningState(toolkit=toolkit, system_prompt=self.system_prompt)
        reasoning_state.update_state_from_input(input, reference_documents)
        logger.debug("Reasoning state updated from input.")
        
        response = self._reason(reasoning_state, toolkit)
        logger.debug("Response generation complete.")
        return response, reasoning_state

    def _reason(self, reasoning_state: ReasoningState, toolkit: Toolkit) -> str:
        logger.debug(f"Entering reasoning loop with current trail length: {len(reasoning_state.trail)}")
        while len(reasoning_state.trail) < 15:
            logger.debug(f"Sending message to LLM provider. Current trail length: {len(reasoning_state.trail)}")
            response = self._send_message(reasoning_state.messages)
            logger.debug(f"Received LLM response: {response}")
            response_type = response.step.type
            logger.debug(f"LLM response type received: {response_type}")
            
            reasoning_state.update_state_from_llm_response(response)
            logger.debug(f"Reasoning state updated from LLM response. Trail length is now: {len(reasoning_state.trail)}")

            if response_type == ReasoningStepType.TOOL_USE_REQUEST:
                logger.debug(f"Tool use request detected for tool: {response.step.tool_name}")
                
                logger.info(f"Invoking tool: {response.step.tool_name} with input: {response.step.input_data}")
                result = toolkit.invoke_tool(response.step.tool_name, response.step.input_data)
                logger.info(f"Tool '{response.step.tool_name}' returned result: {result}")
                logger.debug(f"Tool '{response.step.tool_name}' returned result: {result}")
                
                reasoning_state.add_tool_use_response(result)
                logger.debug("Tool use response added to the reasoning state. Continuing reasoning loop.")
                continue
                
            elif response_type == ReasoningStepType.CLARIFICATION:
                logger.debug(f"Clarification request received: {response.step.clarification}")
                return response.step.clarification
            
            elif response_type == ReasoningStepType.RESPONSE:
                logger.debug(f"Final response obtained: {response.step.response}")
                return response.step.response
            
            else:
                logger.error(f"Unknown response type received from LLM: {response_type}")
                raise ValueError(f"Unknown response type: {response_type}")

        logger.warning("Reasoning loop exceeded maximum allowed iterations without producing a final response.")
        raise RuntimeError("Reasoning loop exceeded maximum iterations without a final response.")

    def _send_message(self, messages: List[dict]) -> Tuple[dict, str]:
        logger.debug(f"Sending messages to LLM provider: {messages}")
        
        if self.provider.supports_streaming:
            logger.debug("Using streaming response generation")
            # Accumulate streamed chunks into full response
            full_response = ""
            for chunk in self.provider.stream_response(
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            ):
                full_response += chunk
            response = full_response
        else:
            logger.debug("Using standard response generation")
            response = self.provider.generate_response(
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
        logger.debug(f"Raw response from LLM provider: {response}")
        try:
            parsed_response = LLMResponse.model_validate_json(response)
            logger.debug(f"Parsed LLM response: {parsed_response}")
        except ValidationError as e:
            logger.error(f"Failed to parse LLM response. Error: {e}. Raw response: {response}")
            raise ValueError(f"Failed to parse LLM response: {e}")
        return parsed_response
